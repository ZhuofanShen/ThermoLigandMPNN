import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from protein_mpnn_utils import parse_PDB


def seq1_index_to_seq2_index(align, index):
    """Do quick conversion of index after alignment"""
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != "-":
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break
    
    # now the index in seq 2 cooresponding to aligned index
    if align.seqB[aln_idx] == "-":
        return None

    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == "-":
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx

def recover_gapped_sequence(occupied_res_seq, original_pdb_indices):
    original_sequence = list(occupied_res_seq)
    index_mapping = dict()
    total_gap_len = 0
    last_seq_idx = len(original_pdb_indices) - 1
    assert last_seq_idx == len(occupied_res_seq) - 1
    for seq_idx, original_pdb_index in enumerate(original_pdb_indices):
        orig_seq_idx = seq_idx + total_gap_len
        index_mapping[orig_seq_idx] = seq_idx
        if seq_idx == last_seq_idx:
            return "".join(original_sequence), index_mapping
        gap_len = original_pdb_indices[seq_idx+1] - original_pdb_index - 1
        if gap_len > 0:
            original_sequence = original_sequence[:orig_seq_idx + 1] + \
                    ["-"] * gap_len + original_sequence[orig_seq_idx + 1:]
            total_gap_len += gap_len
        # elif gap_len < 0:
        #     original_sequence = original_sequence[:orig_seq_idx + 1] + \
        #             ["-"] * 10 + original_sequence[orig_seq_idx + 1:]
        #     total_gap_len += 10


@dataclass
class Mutation:
    position: int
    wildtype: str
    mutation: str
    ddG: Optional[float] = None
    pdb: Optional[str] = str()
    complex: Optional[bool] = False


class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg

        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != "-", :].reset_index(drop=True)
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        if 'reduce' not in cfg:
            cfg.reduce = ''

        self.wt_seqs = {}
        self.mut_rows = {}

        if split == 'all':
            all_names = splits['train'] + splits['val'] + splits['test']
            self.wt_names = all_names
        else:
            if cfg.reduce == 'prot' and split == 'train':
                n_prots_reduced = 58
                self.wt_names = np.random.choice(splits["train"], n_prots_reduced)
            else:
                self.wt_names = splits[split]

        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            if type(cfg.reduce) is float and split == 'train':
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(cfg.reduce), replace=False)

            self.wt_seqs[wt_name] = wt_rows.aa_seq[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        mut_data = self.mut_rows[wt_name]
        wt_seq = self.wt_seqs[wt_name]

        wt_name = wt_name.split(".pdb")[0].replace("|",":")
        pkl_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pkl")
        pdb = pickle.load(open(pkl_file, "rb"))
        assert len(pdb[0]["S"]) == len(wt_seq)
        pdb[0]["S"] = np.array(list("ACDEFGHIKLMNPQRSTVWYX".index(wt_aa) for wt_aa in wt_seq))

        mutations = []
        for i, row in mut_data.iterrows():
            # no insertions, deletions, or double mutants
            if "ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type:
                continue
            assert len(row.aa_seq) == len(wt_seq)
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1
            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut

            if row.ddG_ML == "-":
                continue # filter out any unreliable data

            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)
            mutations.append(Mutation(idx, wt, mut, ddG, wt_name))

        return pdb, mutations


class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):
        self.cfg = cfg

        df = pd.read_csv(self.cfg.data_loc.fireprot_csv).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.fireprot_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.wt_seqs = {}
        self.mut_rows = {}

        if split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.wt_names = all_names
        else:
            self.wt_names = splits[split]

        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query('pdb_id_corrected == @wt_name').reset_index(drop=True)
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        wt_name = self.wt_names[index]
        wt_seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[wt_seq]

        pkl_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{data.pdb_id_corrected[0]}.pkl")
        pdb = pickle.load(open(pkl_file, "rb"))
        occupied_res_seq = "".join("ACDEFGHIKLMNPQRSTVWYX"[aa_onehot] for aa_onehot in pdb[0]['S'])
        full_seq, index_mapping = recover_gapped_sequence(occupied_res_seq, pdb[0]['R_idx'])

        mutations = list()
        for i, row in data.iterrows():
            assert row.pdb_sequence[row.pdb_position] == row.wild_type
            if row.pdb_position < len(full_seq) and full_seq[row.pdb_position] == row.wild_type:
                occupied_res_idx = index_mapping[row.pdb_position]
            else:
                align, *rest = pairwise2.align.globalxx(row.pdb_sequence, full_seq.replace("-", "X"))
                full_res_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if full_res_idx is None:
                    continue
                assert full_seq[full_res_idx] == row.wild_type
                occupied_res_idx = index_mapping[full_res_idx]
            assert occupied_res_seq[occupied_res_idx] == row.wild_type
            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            mut = Mutation(occupied_res_idx, row.wild_type, row.mutation, ddG, wt_name)
            mutations.append(mut)

        return pdb, mutations


class MdrDataset(torch.utils.data.Dataset):

    def __init__(self, mdrdb_tsv, mdrdb_splits, split, mdrdb_pdbs, mut_types, dataset_types):
        df = pd.read_csv(mdrdb_tsv, sep='\t', lineterminator='\n', usecols=["SAMPLE_ID", "DATASET", "TYPE", "UNIPROT_ID", "MUTATION", "DRUG", "SAMPLE_PDB_ID", "DDG_EXP"])
        if "all" not in mut_types:
            query = list()
            for i, mut_type in enumerate(mut_types):
                exec("mut_type_" + str(i) + " = mut_type")
                query.append("TYPE == @mut_type_" + str(i))
            query = " or ".join(query)
            df = df.query(query).reset_index(drop=True)
        if "all" not in dataset_types:
            query = list()
            for i, dataset_type in enumerate(dataset_types):
                exec("dataset_type_" + str(i) + " = dataset_type")
                query.append("DATASET == @dataset_type_" + str(i))
            query = " or ".join(query)
            df = df.query(query).reset_index(drop=True)
        df['PROTEIN_DRUG'] = df.apply(lambda row: row['UNIPROT_ID'] + "_" + row['DRUG'], axis=1)

        with open(mdrdb_splits, 'rb') as f:
            split_uniprot_drug_dict = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.pdb_dir = mdrdb_pdbs

        if split == 'all':
            split_list = ['train', 'val', 'test']
        else:
            split_list = [split]
        
        self.uniprot_drug_df_dict = dict()
        self.uniprot_drug_pdb_id_list = list()
        for split in split_list:
            uniprot_drug_pdb_ids_dict = split_uniprot_drug_dict[split]
            for uniprot_drug, pdb_ids in uniprot_drug_pdb_ids_dict.items():
                self.uniprot_drug_df_dict[uniprot_drug] = df.query('PROTEIN_DRUG == @uniprot_drug').reset_index(drop=True)
                for pdb_id in pdb_ids[1:]:
                    self.uniprot_drug_pdb_id_list.append((uniprot_drug, pdb_id))

        self.align_uniprot_pdb_ids_dict = {"P00533": ["1MOX", "7SZ0"], "P00519": ["5DC4", "2V7A", "6XR6", "7N9G"], "P36897": ["3HMM", "6B8Y"], "P10415": ["4IEH", "2O21"], "P01137": ["5FFO", "5VQP", "4KV5"], "P00374": ["5SD8"], "P05067": ["3UMI"], "P08069":["6VWG", "6VWJ", "5U8R"], "Q13564": ["3GZN"]}
        self.align_uniprot_sequences_dict = {
            "P00533": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA", 
            "P00519": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR", 
            "P36897": "MEAAVAAPRPRLLLLVLAAAAAAAAALLPGATALQCFCHLCTKDNFTCVTDGLCFVSVTETTDKVIHNSMCIAEIDLIPRDRPFVCAPSSKTGSVTTTYCCNQDHCNKIELPTTVKSSPGLGPVELAAVIAGPVCFVCISLMLMVYICHNRTVIHHRVPNEEDPSLDRPFISEGTTLKDLIYDMTTSGSGSGLPLLVQRTIARTIVLQESIGKGRFGEVWRGKWRGEEVAVKIFSSREERSWFREAEIYQTVMLRHENILGFIAADNKDNGTWTQLWLVSDYHEHGSLFDYLNRYTVTVEGMIKLALSTASGLAHLHMEIVGTQGKPAIAHRDLKSKNILVKKNGTCCIADLGLAVRHDSATDTIDIAPNHRVGTKRYMAPEVLDDSINMKHFESFKRADIYAMGLVFWEIARRCSIGGIHEDYQLPYYDLVPSDPSVEEMRKVVCEQKLRPNIPNRWQSCEALRVMAKIMRECWYANGAARLTALRIKKTLSQLSQQEGIKM", 
            "P10415": "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK", 
            "P01137": "MPPSGLRLLPLLLPLLWLLVLTPGRPAAGLSTCKTIDMELVKRKRIEAIRGQILSKLRLASPPSQGEVPPGPLPEAVLALYNSTRDRVAGESAEPEPEPEADYYAKEVTRVLMVETHNEIYDKFKQSTHSIYMFFNTSELREAVPEPVLLSRAELRLLRLKLKVEQHVELYQKYSNNSWRYLSNRLLAPSDSPEWLSFDVTGVVRQWLSRGGEIEGFRLSAHCSCDSRDNTLQVDINGFTTGRRGDLATIHGMNRPFLLLMATPLERAQHLQSSRHRRALDTNYCFSSTEKNCCVRQLYIDFRKDLGWKWIHEPKGYHANFCLGPCPYIWSLDTQYSKVLALYNQHNPGASAAPCCVPQALEPLPIVYYVGRKPKVEQLSNMIVRSCKCS", 
            "P00374": "VGSLNCIVAVSQNMGIGKNGDLPWPPLRNEFRYFQRMTTTSSVEGKQNLVIMGKKTWFSIPEKNRPLKGRINLVLSRELKEPPQGAHFLSRSLDDALKLTEQPELANKVDMVWIVGGSSVYKEAMNHPGHLKLFVTRIMQDFESDTFFPEIDLEKYKLLPEYPGVLSDVQEEKGIKYKFEVYEKND", 
            "P05067": "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN", 
            "P08069": "MKSGSGGGSPTSLWGLLFLSAALSLWPTSGEICGPGIDIRNDYQQLKRLENCTVIEGYLHILLISKAEDYRSYRFPKLTVITEYLLLFRVAGLESLGDLFPNLTVIRGWKLFYNYALVIFEMTNLKDIGLYNLRNITRGAIRIEKNADLCYLSTVDWSLILDAVSNNYIVGNKPPKECGDLCPGTMEEKPMCEKTTINNEYNYRCWTTNRCQKMCPSTCGKRACTENNECCHPECLGSCSAPDNDTACVACRHYYYAGVCVPACPPNTYRFEGWRCVDRDFCANILSAESSDSEGFVIHDGECMQECPSGFIRNGSQSMYCIPCEGPCPKVCEEEKKTKTIDSVTSAQMLQGCTIFKGNLLINIRRGNNIASELENFMGLIEVVTGYVKIRHSHALVSLSFLKNLRLILGEEQLEGNYSFYVLDNQNLQQLWDWDHRNLTIKAGKMYFAFNPKLCVSEIYRMEEVTGTKGRQSKGDINTRNNGERASCESDVLHFTSTTTSKNRIIITWHRYRPPDYRDLISFTVYYKEAPFKNVTEYDGQDACGSNSWNMVDVDLPPNKDVEPGILLHGLKPWTQYAVYVKAVTLTMVENDHIRGAKSEILYIRTNASVPSIPLDVLSASNSSSQLIVKWNPPSLPNGNLSYYIVRWQRQPQDGYLYRHNYCSKDKIPIRKYADGTIDIEEVTENPKTEVCGGEKGPCCACPKTEAEKQAEKEEAEYRKVFENFLHNSIFVPRPERKRRDVMQVANTTMSSRSRNTTAADTYNITDPEELETEYPFFESRVDNKERTVISNLRPFTLYRIDIHSCNHEAEKLGCSASNFVFARTMPAEGADDIPGPVTWEPRPENSIFLKWPEPENPNGLILMYEIKYGSQVEDQRECVSRQEYRKYGGAKLNRLNPGNYTARIQATSLSGNGSWTDPVFFYVQAKTGYENFIHLIIALPVAVLLIVGGLVIMLYVFHRKRNNSRLGNGVLYASVNPEYFSAADVYVPDEWEVAREKITMSRELGQGSFGMVYEGVAKGVVKDEPETRVAIKTVNEAASMRERIEFLNEASVMKEFNCHHVVRLLGVVSQGQPTLVIMELMTRGDLKSYLRSLRPEMENNPVLAPPSLSKMIQMAGEIADGMAYLNANKFVHRDLAARNCMVAEDFTVKIGDFGMTRDIYETDYYRKGGKGLLPVRWMSPESLKDGVFTTYSDVWSFGVVLWEIATLAEQPYQGLSNEQVLRFVMEGGLLDKPDNCPDMLFELMRMCWQYNPKMRPSFLEIISSIKEEMEPGFREVSFYYSEENKLPEPEELDLEPENMESVPLDPSASSSSLPLPDRHSGHKAENGPGPGVLVLRASFDERQPYAHMNGGRKNERALPLPQSSTC", 
            "Q13564": "MAQLGKLLKEQKYDRQLRLWGDHGQEALESAHVCLINATATGTEILKNLVLPGIGSFTIIDGNQVSGEDAGNNFFLQRSSIGKNRAEAAMEFLQELNSDVSGSFVEESPENLLDNDPSFFCRFTVVVATQLPESTSLRLADVLWNSQIPLLICRTYGLVGYMRIIIKEHPVIESHPDNALEDLRLDKPFPELREHFQSYDLDHMEKKDHSHTPWIVIIAKYLAQWYSETNGRIPKTYKEKEDFRDLIRQGILKNENGAPEDEENFEEAIKNVNTALNTTQIPSSIEDIFNDDRCINITKQTPSFWILARALKEFVAKEGQGNLPVRGTIPDMIADSGKYIKLQNVYREKAKKDAAAVGNHVAKLLQSIGQAPESISEKELKLLCSNSAFLRVVRCRSLAEEYGLDTINKDEIISSMDNPDNEIVLYLMLRAVDRFHKQQGRYPGVSNYQVEEDIGKLKSCLTGFLQEYGLSVMVKDDYVHEFCRYGAAEPHTIAAFLGGAAAQEVIKIITKQFVIFNNTYIYSGMSQTSATFQL"
        }

    def __len__(self):
        return len(self.uniprot_drug_pdb_id_list)

    def __getitem__(self, index):
        uniprot_drug, pdb_id = self.uniprot_drug_pdb_id_list[index]
        uniprot_drug_df = self.uniprot_drug_df_dict[uniprot_drug]

        pdb_drug_df = uniprot_drug_df.query('SAMPLE_PDB_ID == @pdb_id').reset_index(drop=True)
        sample_id = pdb_drug_df["SAMPLE_ID"][0]
        mutation_type = pdb_drug_df["TYPE"][0].replace(" ", "_")
        uniprot = pdb_drug_df["UNIPROT_ID"][0]
        # pkl_file = os.path.join(self.pdb_dir, mutation_type, sample_id, "WT_" + pdb_id + '_complex.pkl')
        # pdb = pickle.load(open(pkl_file, "rb"))
        pdb_file = os.path.join(self.pdb_dir, mutation_type, sample_id, "WT_" + pdb_id + '_complex.pdb')
        pdb = parse_PDB(pdb_file)
        pdb_numbers = list(str(int(pdb_number)) for pdb_number in pdb[0]["R_idx"])
        incorrect_pdb_indices_pdb_ids = self.align_uniprot_pdb_ids_dict.get(uniprot)
        align = None
        if incorrect_pdb_indices_pdb_ids is not None and pdb_id in incorrect_pdb_indices_pdb_ids:
            uniprot_seq = self.align_uniprot_sequences_dict[uniprot]
            occupied_res_seq = "".join("ACDEFGHIKLMNPQRSTVWYX"[aa_onehot] for aa_onehot in pdb[0]['S'])
            full_seq, index_mapping = recover_gapped_sequence(occupied_res_seq, pdb[0]['R_idx'])
            align, *rest = pairwise2.align.globalxx(uniprot_seq, full_seq.replace("-", "X"))

        mut_ddG_dict = dict()
        for mut in uniprot_drug_df.MUTATION.unique():
            mut_df = uniprot_drug_df.query('MUTATION == @mut').reset_index(drop=True)
            ddG_values = mut_df.DDG_EXP.unique()
            mut_ddG_dict[mut] = sum(ddG_values) / len(ddG_values)

        mutations = list()
        for mut, ddG in mut_ddG_dict.items():
            wtAA, pdb_number, mutAA = mut[0], mut[1:-1], mut[-1]
            if align is None:
                try:
                    occupied_res_idx = pdb_numbers.index(pdb_number)
                except ValueError:
                    continue
                try:
                    assert "ACDEFGHIKLMNPQRSTVWYX"[pdb[0]['S'][occupied_res_idx]] == wtAA
                except AssertionError:
                    continue
            else:
                uniprot_res_idx = int(pdb_number) - 1
                if uniprot_seq[uniprot_res_idx] != wtAA:
                    continue
                full_res_idx = seq1_index_to_seq2_index(align, uniprot_res_idx)
                if full_res_idx is None:
                    continue
                occupied_res_idx = index_mapping[full_res_idx]
                assert occupied_res_seq[occupied_res_idx] == wtAA
            ddG = torch.tensor([ddG], dtype=torch.float32)
            mutations.append(Mutation(occupied_res_idx, wtAA, mutAA, ddG, pdb_id, True))

        return pdb, mutations


class ddgBenchDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname):

        self.cfg = cfg
        self.pdb_dir = pdb_dir

        df = pd.read_csv(csv_fname)

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()

        for wt_name in self.wt_names:
            wt_name_query = wt_name
            wt_name = wt_name[:-1]
            self.mut_rows[wt_name] = df.query('PDB == @wt_name_query').reset_index(drop=True)
            if 'S669' not in self.pdb_dir:
                self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = [wt_name[-1]]

        wt_name = wt_name.split(".pdb")[0][:-1]
        mut_data = self.mut_rows[wt_name]

        pdb_file = os.path.join(self.pdb_dir, wt_name + '.pdb')

        # modified PDB parser returns list of residue IDs so we can align them easier
        pdb = parse_PDB(pdb_file, chain)
        pdb_numbers = list(str(int(resn)) for resn in pdb[0]["R_idx"])

        mutations = list()
        for i, row in mut_data.iterrows():
            mut = row.MUT
            wtAA, pdb_number, mutAA = mut[0], mut[1:-1], mut[-1]
            try:
                occupied_res_idx = pdb_numbers.index(pdb_number)
            except ValueError:  # skip positions with insertion codes for now - hard to parse
                continue
            try:
                assert "ACDEFGHIKLMNPQRSTVWYX"[pdb[0]['S'][occupied_res_idx]] == wtAA
            except AssertionError:  # contingency for mis-alignments
                continue
            ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor([row.DDG * -1.], dtype=torch.float32)
            mutations.append(Mutation(occupied_res_idx, wtAA, mutAA, ddG, wt_name))

        return pdb, mutations


class ComboDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        datasets = []
        if "fireprot" in cfg.datasets:
            fireprot = FireProtDataset(cfg, split)
            datasets.append(fireprot)
        if "megascale" in cfg.datasets:
            mega_scale = MegaScaleDataset(cfg, split)
            datasets.append(mega_scale)
        if "mdrdb" in cfg.datasets:
            # mdrdb = MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits, split, cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"])
            mdrdb = MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits, split, cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["AIMMS", "Platinum", "TKI"])
            datasets.append(mdrdb)
        self.mut_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.mut_dataset)

    def __getitem__(self, index):
        return self.mut_dataset[index]
