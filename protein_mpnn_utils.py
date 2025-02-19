import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prody import *


confProDy(verbosity="none")

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
element_dict = dict(zip(element_list, range(1, len(element_list))))

restype_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

atom_order = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "CB": 3,
    "O": 4,
    "CG": 5,
    "CG1": 6,
    "CG2": 7,
    "OG": 8,
    "OG1": 9,
    "SG": 10,
    "CD": 11,
    "CD1": 12,
    "CD2": 13,
    "ND1": 14,
    "ND2": 15,
    "OD1": 16,
    "OD2": 17,
    "SD": 18,
    "CE": 19,
    "CE1": 20,
    "CE2": 21,
    "CE3": 22,
    "NE": 23,
    "NE1": 24,
    "NE2": 25,
    "OE1": 26,
    "OE2": 27,
    "CH2": 28,
    "NH1": 29,
    "NH2": 30,
    "OH": 31,
    "CZ": 32,
    "CZ2": 33,
    "CZ3": 34,
    "NZ": 35,
    "OXT": 36,
}

atom_types_4 = ["N", "CA", "C", "O"]

atom_types_all = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
]


def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m

def parse_PDB(
    input_path: str,
    chains: list = None,
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """

    if not parse_all_atoms:
        atom_types = atom_types_4
    else:
        atom_types = atom_types_all
    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains and len(chains) > 0:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])

    protein_atoms = atoms.select("protein")
    # backbone = protein_atoms.select("backbone")
    # other_atoms = atoms.select("not protein and not water")
    # water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = dict()
    chain_mask = list()
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i
        chain_mask.append(1 if not chains or CA_chain_ids[i] in chains else 0)

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    # b = CA - N
    # c = C - CA
    # a = np.cross(b, c, axis=-1)
    # CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    seq = CA_atoms.getResnames()
    seq = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(seq)]
    S = np.array(["ACDEFGHIKLMNPQRSTVWYX".find(AA) for AA in seq], dtype=np.int32) # restype_STRtoINT[AA]
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    Y = list()
    Y_t = list()
    for line in open(input_path, "r"):
        if ((line.startswith("ATOM  ") or line.startswith("HETATM")) and \
                not line[17:20] in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", \
                                    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", \
                                    "UNK", "MSE", "PTR", "CYX", "HID", "HIE", "HIP", \
                                    "EOH", "EDO", "GOL", "PEG", "PG4", "PG5", "1PE", "PG6", \
                                    "ACN", "FMT", "ACT", " NA", " MG", "  K", " CA", " CL", "SO4", "PO4", "WAT", "HOH"] \
                or line.startswith("HETATM") and line[17:20] == "UNK") \
                and len(line) >= 78:
            if chains and line[20:22].strip() not in chains:
                continue
            atom_type = line.rstrip()[-4:].strip().upper()
            if atom_type.endswith("+") or atom_type.endswith("-"):
                atom_type = atom_type[:-2]
            atom_type = element_dict[atom_type]
            if atom_type == 1:
                continue
            Y.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            Y_t.append(atom_type)
    Y = np.array(Y, dtype=np.float32)
    Y_t = np.array(Y_t, dtype=np.int32)
    Y_m = (Y_t != 1) * (Y_t != 0)

    output_dict = dict()
    output_dict["name"] = input_path[(input_path.rfind("/") + 1):-4]
    output_dict["X"] = X
    output_dict["mask"] = np.array(mask, dtype=np.int32)
    output_dict["R_idx"] = R_idx
    output_dict["chain_labels"] = chain_labels

    # output_dict["chain_letters"] = CA_chain_ids

    # mask_c = list()
    # chain_list = list(set(CA_chain_ids))
    # chain_list.sort()
    # for chain in chain_list:
    #     mask_c.append([chain == item for item in CA_chain_ids])

    # output_dict["mask_c"] = mask_c
    # output_dict["chain_list"] = chain_list
    output_dict['chain_mask'] = np.array(chain_mask, dtype=np.int32)
    output_dict["S"] = S
    output_dict["xyz_37"] = xyz_37
    output_dict["xyz_37_m"] = xyz_37_m

    output_dict["Y"] = Y
    output_dict["Y_t"] = Y_t
    output_dict["Y_m"] = Y_m

    return [output_dict]

def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="ligand_mpnn",
    device="cuda"
):
    output_dict = {}
    if model_type.startswith("ligand_mpnn"):
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = np.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
        L2_AB = np.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

        nn_idx = np.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
        L2_AB_nn = np.take_along_axis(L2_AB, nn_idx, axis=1)
        D_XY = np.sqrt(L2_AB_nn[:, 0])

        Y_r = np.tile(Y, (CB.shape[0], 1, 1)) # Y[None, :, :].repeat(CB.shape[0], 1, 1)
        Y_t_r = np.tile(Y_t, (CB.shape[0], 1)) # Y_t[None, :].repeat(CB.shape[0], 1)
        Y_m_r = np.tile(Y_m, (CB.shape[0], 1)) # Y_m[None, :].repeat(CB.shape[0], 1)

        Y = np.take_along_axis(Y_r, np.tile(nn_idx[:,:,None], (1, 1, 3)), axis=1)
        Y_t = np.take_along_axis(Y_t_r, nn_idx, axis=1)
        Y_m = np.take_along_axis(Y_m_r, nn_idx, axis=1)
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]

        if model_type == "ligand_mpnn_new":
            nn_idx_torch = torch.tensor(nn_idx, dtype=torch.int64, device=device)
            Y_mask_scattered = torch.zeros([input_dict["S"].shape[0], input_dict["Y_t"].shape[0]], dtype=torch.int32, device=device) # [L, l]
            Y_scale_unsqueezed = torch.sum(Y_mask_scattered.scatter_(1, nn_idx_torch, 
                                 torch.ones([nn_idx.shape[0], nn_idx.shape[1]], dtype=torch.int32, device=device)), dim=0) # [l]
            # squeeze ligand atoms without neighborhood residues
            nn_idx_squeezed_scattered = [-1] * input_dict["Y_t"].shape[0]
            Y_scale_squeezed = list()
            for i_atom, scale in enumerate(Y_scale_unsqueezed):
                if scale > 0:
                    nn_idx_squeezed_scattered[i_atom] = len(Y_scale_squeezed)
                    Y_scale_squeezed.append(scale)
            nn_idx_squeezed_scattered = torch.tensor(nn_idx_squeezed_scattered, dtype=torch.int64, device=device)
            # nn_idx_squeezed = gather_context_atoms(nn_idx_squeezed_scattered, nn_idx)
            nn_idx_squeezed_scattered_r = nn_idx_squeezed_scattered[None, :].repeat(nn_idx.shape[0], 1) # [L, l]
            nn_idx_squeezed_scattered_tmp = torch.gather(nn_idx_squeezed_scattered_r, 1, nn_idx_torch)
            nn_idx_squeezed = torch.zeros(
                [nn_idx.shape[0], nn_idx.shape[1]], dtype=torch.int64, device=device
            )
            nn_idx_squeezed[:, :nn_idx_squeezed_scattered_tmp.shape[1]] = nn_idx_squeezed_scattered_tmp
            Y_scale_squeezed = torch.tensor(Y_scale_squeezed, dtype=torch.int32, device=device)
            output_dict["nn_idx"] = nn_idx_squeezed[None,] #[B,L,M]
            output_dict["Y_scale"] = Y_scale_squeezed[None,] #[B,l]

        output_dict["mask_XY"] = torch.from_numpy(mask_XY[None,]).to(dtype=torch.int32, device=device)
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = torch.from_numpy(input_dict["side_chain_mask"][None,]).to(dtype=torch.int32, device=device)
        output_dict["Y"] = torch.from_numpy(Y[None,]).to(dtype=torch.float32, device=device)
        output_dict["Y_t"] = torch.from_numpy(Y_t[None,]).to(dtype=torch.int32, device=device)
        output_dict["Y_m"] = torch.from_numpy(Y_m[None,]).to(dtype=torch.int32, device=device)
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx

    output_dict["R_idx"] = torch.from_numpy(np.array(R_idx_list, dtype=np.int32)[None,]).to(dtype=torch.long, device=device)
    output_dict["R_idx_original"] = torch.from_numpy(input_dict["R_idx"][None,]).to(dtype=torch.long, device=device)
    output_dict["chain_labels"] = torch.from_numpy(input_dict["chain_labels"][None,]).to(dtype=torch.int32, device=device)
    output_dict["S"] = torch.from_numpy(input_dict["S"][None,]).to(dtype=torch.int32, device=device)
    output_dict["chain_mask"] = torch.from_numpy(input_dict["chain_mask"][None,]).to(dtype=torch.int32, device=device)
    output_dict["mask"] = torch.from_numpy(input_dict["mask"][None,]).to(dtype=torch.int32, device=device)

    output_dict["X"] = torch.from_numpy(input_dict["X"][None,]).to(dtype=torch.float32, device=device)

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = torch.from_numpy(input_dict["xyz_37"][None,]).to(dtype=torch.float32, device=device)
        output_dict["xyz_37_m"] = torch.from_numpy(input_dict["xyz_37_m"][None,]).to(dtype=torch.int32, device=device)

    return output_dict

# gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def gather_context_atom_features(Y, nn_idx):
    # Y [B, l, C] at Neighbor indices [B, L, M] => Y [B, L, M, C]
    Y_r = Y[:, None, :, :].repeat(1, nn_idx.shape[1], 1, 1) # [B, L, l, C]
    Y_tmp = torch.gather(Y_r, 2, nn_idx[:, :, :, None].repeat(1, 1, 1, Y.shape[-1]))
    Y = torch.zeros(
        [nn_idx.shape[0], nn_idx.shape[1], nn_idx.shape[2], Y.shape[-1]], 
        dtype=torch.float32, device=Y.device
    )
    Y[:, :, :Y_tmp.shape[2]] = Y_tmp
    return Y


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        device=None,
        atom_context_num=0,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=False,
    ):
        super(ProteinMPNN, self).__init__()

        self.model_type = model_type
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        
        if self.model_type == "protein_mpnn":
            self.features = ProteinFeatures(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps
            )

        elif self.model_type.startswith("ligand_mpnn"):
            self.features = ProteinFeaturesLigand(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                device=device,
                atom_context_num=atom_context_num,
                use_side_chains=ligand_mpnn_use_side_chain_context,
            )

            self.W_nodes_y = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.W_edges_y = nn.Linear(hidden_dim, hidden_dim, bias=True)

            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
            self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)

            if self.model_type == "ligand_mpnn":
                self.y_context_encoder_layers = nn.ModuleList(
                    [DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)]
                )
                self.context_encoder_layers = nn.ModuleList(
                    [
                        DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                        for _ in range(2)
                    ]
                )

                self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.V_C_norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout)

            elif self.model_type == "ligand_mpnn_new":
                self.y_context_encoder_layer_1 = DecLayerJ(hidden_dim, hidden_dim, dropout=dropout)
                self.context_encoder_layer_1 = DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout)

                self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.V_C_norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout)

                self.protein2ligandlayer = Protein2LigandLayer(hidden_dim, hidden_dim * 2, dropout=dropout)

                self.y_context_encoder_layer_2 = DecLayerJ(hidden_dim, hidden_dim, dropout=dropout)
                self.context_encoder_layer_2 = DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout)

                self.V_C_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.V_C_norm_2 = nn.LayerNorm(hidden_dim)
                self.dropout_2 = nn.Dropout(dropout)

        # Decoder layers
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def protein_encode(self, E, E_idx, mask):
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device) #[B,L,C]
        h_E = self.W_e(E) #[B,L,K,C]

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend #[B,L,M,M,C]
        for encoder_layer in self.encoder_layers:
            h_V, h_E = encoder_layer(h_V, h_E, E_idx, mask, mask_attend) # ([B,L,C], [B,L,K,C]) --> ([B,L,C], [B,L,K,C])
        return h_V, h_E

    def protein_ligand_encode(self, Y_nodes, Y_edges, Y_m, h_V, E_context, mask):
        h_Y_nodes = self.W_nodes_y(Y_nodes) #[B,L,M,C]
        h_Y_edges = self.W_edges_y(Y_edges) #[B,L,M,M,C]
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :] #[B,L,M,M,C]
        h_E_context = self.W_v(E_context) #[B,L,M,C], protein-ligand edges
        h_V_C = self.W_c(h_V) #[B,L,C]
        for y_context_encoder_layer, context_encoder_layer in \
                zip(self.y_context_encoder_layers, self.context_encoder_layers):

            # ligand graph: neighborhood ligand nodes & edges --> update central ligand nodes
            h_Y_nodes = y_context_encoder_layer(h_Y_nodes, h_Y_edges, Y_m, Y_m_edges) # ([B,L,M,C], [B,L,M,M,C]) --> [B,L,M,C]

            # protein-ligand graph: neighborhood ligand nodes --> update central residue nodes
            h_E_context_Y_nodes = torch.cat([h_E_context, h_Y_nodes], -1) # [B,L,M,2C]
            h_V_C = context_encoder_layer(h_V_C, h_E_context_Y_nodes, mask, Y_m) # ([B,L,C], [B,L,M,2C]) --> [B,L,C]

        h_V_C = self.V_C(h_V_C) # [B,L,C]
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C)) # [B,L,C]

        return h_V

    def protein_ligand_bidirectional_encode(self, Y_nodes, Y_edges, Y_m, h_V, E_context, mask, nn_idx, Y_scale):
        h_Y_nodes = self.W_nodes_y(Y_nodes) #[B,L,M,C]
        h_Y_edges = self.W_edges_y(Y_edges) #[B,L,M,M,C]
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :] #[B,L,M,M,C]
        h_E_context = self.W_v(E_context) #[B,L,M,C], protein-ligand edges
        h_V_C = self.W_c(h_V) #[B,L,C]
        # ligand graph: neighborhood ligand nodes & edges --> update central ligand nodes
        h_Y_nodes = self.y_context_encoder_layer_1(h_Y_nodes, h_Y_edges, Y_m, Y_m_edges) # ([B,L,M,C], [B,L,M,M,C]) --> [B,L,M,C]

        # protein-ligand graph: neighborhood ligand nodes --> update central residue nodes
        h_E_context_Y_nodes = torch.cat([h_E_context, h_Y_nodes], -1) # [B,L,M,2C]
        h_V_C = self.context_encoder_layer_1(h_V_C, h_E_context_Y_nodes, mask, Y_m) # ([B,L,C], [B,L,M,2C]) --> [B,L,C]

        h_V_C = self.V_C(h_V_C) # [B,L,C]
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C)) # [B,L,C]

        # protein-ligand graph: update ligand nodes and edges
        h_Y_nodes, h_E_context = self.protein2ligandlayer(nn_idx, Y_scale, h_Y_nodes, h_E_context, h_V, mask, Y_m)

        # ligand graph: neighborhood ligand nodes & edges --> update central ligand nodes
        h_Y_nodes = self.y_context_encoder_layer_2(h_Y_nodes, h_Y_edges, Y_m, Y_m_edges) # ([B,L,M,C], [B,L,M,M,C]) --> [B,L,M,C]

        # protein-ligand graph: neighborhood ligand nodes --> update central residue nodes
        h_E_context_Y_nodes = torch.cat([h_E_context, h_Y_nodes], -1) # [B,L,M,2C]
        h_V_C = self.context_encoder_layer_2(h_V_C, h_E_context_Y_nodes, mask, Y_m) # ([B,L,C], [B,L,M,2C]) --> [B,L,C]

        h_V_C = self.V_C_2(h_V_C) # [B,L,C]
        h_V = h_V + self.V_C_norm_2(self.dropout_2(h_V_C)) # [B,L,C]

        return h_V

    def decode(self, h_S, h_V, h_E, E_idx, decode_mask, mask):
        mask_1D = mask.view([h_S.shape[0], h_S.shape[1], 1, 1])
        mask_bw = mask_1D * decode_mask
        mask_fw = mask_1D * (1.0 - decode_mask)

        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder # [B,L,K,3C]

        all_hidden = list() # for ThermoMPNN
        for decoder_layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = decoder_layer(h_V, h_ESV, mask)
            all_hidden.append(h_V)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        return list(reversed(all_hidden)), log_probs

    def forward(self, feature_dict, complex=False):
        # xyz_37 = feature_dict["xyz_37"] #[B,L,37,3] - xyz coordinates for all atoms if needed
        # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        # Y = feature_dict["Y"] #[B,L,30,3] - for ligandMPNN coords
        # Y_t = feature_dict["Y_t"] #[B,L,30] - element type
        # Y_m = feature_dict["Y_m"] #[B,L,30] - mask
        # X = feature_dict["X"] #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
        S = feature_dict["S"] # [B,L] - integer proitein sequence encoded using "restype_STRtoINT"
        mask = feature_dict["mask"] # [B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict["chain_mask"] # [B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed

        B, L = S.shape
        device = S.device

        if self.model_type.startswith("ligand_mpnn"):
            E_context, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(feature_dict)
        else:
            E, E_idx = self.features(feature_dict)
        # V:[B,L,M,C] E:[B,L,K,C] E_idx:[B,L,K] Y_nodes:[B,L,M,C] Y_edges:[B,L,M,M,C] Y_m:[B,L,M]

        h_V, h_E = self.protein_encode(E, E_idx, mask)

        if self.model_type == "ligand_mpnn":
            h_V_ctxt = self.protein_ligand_encode(Y_nodes, Y_edges, Y_m, h_V, E_context, mask)
        elif self.model_type == "ligand_mpnn_new":
            nn_idx = feature_dict["nn_idx"]
            Y_scale = feature_dict["Y_scale"]
            if Y_scale.shape[1] > 0:
                h_V_ctxt = self.protein_ligand_bidirectional_encode(Y_nodes, Y_edges, Y_m, h_V, E_context, mask, nn_idx, Y_scale)
            else:
                h_V_ctxt = h_V

        chain_mask = mask * chain_mask  # update chain_M to include missing regions
        # default ProteinMPNN random autoregressive decoding
        # numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0
        # decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(torch.randn(chain_mask.shape, device=device))))
        # for ThermoMPNN, decode left-to-right with all residues visible - we know the surrounding sequence
        decoding_order = torch.tensor([list(range(L))], device=device)
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
        # 0 (invisible) for current AA and any AAs decoded BEFORE it
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(L, L, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        ) # [B, L, L] array of visibility ordered backward
        # for ThermoMPNN, set all residues to be visible
        order_mask_backward = torch.ones_like(order_mask_backward)
        decode_mask = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

        h_S = self.W_s(S)
        mpnn_hid = None
        log_probs = None
        mpnn_hid_ctxt = None
        log_probs_ctxt = None
        if self.model_type == "protein_mpnn" or self.model_type.startswith("ligand_mpnn") and complex:
            mpnn_hid, log_probs = self.decode(h_S, h_V, h_E, E_idx, decode_mask, mask)
        if self.model_type.startswith("ligand_mpnn"):
            mpnn_hid_ctxt, log_probs_ctxt = self.decode(h_S, h_V_ctxt, h_E, E_idx, decode_mask, mask)
        return h_S, mpnn_hid, log_probs, mpnn_hid_ctxt, log_probs_ctxt


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeaturesLigand(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        device=None,
        atom_context_num=16,
        use_side_chains=False,
    ):
        """Extract protein features"""
        super(ProteinFeaturesLigand, self).__init__()

        self.use_side_chains = use_side_chains

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

        self.node_project_down = nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = nn.LayerNorm(node_features)

        self.type_linear = nn.Linear(147, 64)

        self.y_nodes = nn.Linear(147, node_features, bias=False)
        self.y_edges = nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = nn.LayerNorm(node_features)
        self.norm_y_nodes = nn.LayerNorm(node_features)

        self.atom_context_num = atom_context_num

        # the last 32 atoms in the 37 atom representation
        self.side_chain_atom_types = torch.tensor(side_chain_atom_types, device=device)
        self.periodic_table_features = torch.tensor(periodic_table_features, dtype=torch.long, device=device)

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = F.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = F.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        Y = input_features["Y"] # [B, L, 30, 3]
        Y_m = input_features["Y_m"] # [B, L, 30]
        Y_t = input_features["Y_t"] # [B, L, 30]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + self.augment_eps * torch.randn_like(Y)

        B, L, _, _ = X.shape

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        if self.use_side_chains:
            xyz_37 = input_features["xyz_37"]
            xyz_37_m = input_features["xyz_37_m"]
            E_idx_sub = E_idx[:, :, :16]  # [B, L, 15]
            mask_residues = input_features["chain_mask"]
            xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])
            R_m = gather_nodes(xyz_37_m[:, :, 5:], E_idx_sub)

            X_sidechain = xyz_37[:, :, 5:, :].view(B, L, -1)
            R = gather_nodes(X_sidechain, E_idx_sub).view(
                B, L, E_idx_sub.shape[2], -1, 3
            )
            R_t = self.side_chain_atom_types[None, None, None, :].repeat(
                B, L, E_idx_sub.shape[2], 1
            )

            # Side chain atom context
            R = R.view(B, L, -1, 3)  # coordinates
            R_m = R_m.view(B, L, -1)  # mask
            R_t = R_t.view(B, L, -1)  # atom types

            # Ligand atom context
            Y = torch.cat((R, Y), 2)  # [B, L, atoms, 3]
            Y_m = torch.cat((R_m, Y_m), 2)  # [B, L, atoms]
            Y_t = torch.cat((R_t, Y_t), 2)  # [B, L, atoms]

            Cb_Y_distances = torch.sum((Cb[:, :, None, :] - Y) ** 2, -1)
            mask_Y = mask[:, :, None] * Y_m
            Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
            _, E_idx_Y = torch.topk(
                Cb_Y_distances_adjusted, self.atom_context_num, dim=-1, largest=False
            )

            Y = torch.gather(Y, 2, E_idx_Y[:, :, :, None].repeat(1, 1, 1, 3))
            Y_t = torch.gather(Y_t, 2, E_idx_Y)
            Y_m = torch.gather(Y_m, 2, E_idx_Y)

        Y_t = Y_t.long()
        Y_t_g = self.periodic_table_features[1][Y_t]  # group; 19 categories including 0
        Y_t_p = self.periodic_table_features[2][Y_t]  # period; 8 categories including 0

        Y_t_g_1hot_ = F.one_hot(Y_t_g, 19)  # [B, L, M, 19]
        Y_t_p_1hot_ = F.one_hot(Y_t_p, 8)  # [B, L, M, 8]
        Y_t_1hot_ = F.one_hot(Y_t, 120)  # [B, L, M, 120]

        Y_t_1hot_ = torch.cat(
            [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1
        )  # [B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_.float())

        D_N_Y = self._rbf(
            torch.sqrt(torch.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )  # [B, L, M, num_bins]
        D_Ca_Y = self._rbf(
            torch.sqrt(torch.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )
        D_C_Y = self._rbf(torch.sqrt(torch.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_O_Y = self._rbf(torch.sqrt(torch.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6))
        D_Cb_Y = self._rbf(
            torch.sqrt(torch.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6)
        )

        f_angles = self._make_angle_features(N, Ca, C, Y)  # [B, L, M, 4]

        D_all = torch.cat(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1
        )  # [B,L,M,5*num_bins+5]
        E_context = self.node_project_down(D_all)  # [B, L, M, node_features]
        E_context = self.norm_nodes(E_context)

        Y_edges = self._rbf(
            torch.sqrt(
                torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )  # [B, L, M, M, num_bins]

        Y_edges = self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_.float())

        Y_edges = self.norm_y_edges(Y_edges)
        Y_nodes = self.norm_y_nodes(Y_nodes)

        return E_context, E, E_idx, Y_nodes, Y_edges, Y_m


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(
            -1, -1, -1, h_E.size(-2), -1
        )  # the only difference
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class Protein2LigandLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(Protein2LigandLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, nn_idx, Y_scale, h_Y_nodes, h_E_context, h_V, mask_V=None, mask_attend=None):
        # central residue nodes --> update neighborhood ligand nodes
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_Y_nodes.size(-2), -1)
        h_YEV = torch.cat([h_Y_nodes, h_E_context, h_V_expand], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_YEV))))) # [B,L,M,C]
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        # average messages passing to h_Y_nodes representing each same ligand atom
        # dh = torch.sum(h_message, -2) / self.scale
        h_message_scattered = torch.zeros(
            [h_message.shape[0], h_message.shape[1], Y_scale.shape[1], h_message.shape[3]], 
            dtype=h_message.dtype, device=h_Y_nodes.device
        ) # [B,L,l,C]
        h_message_scattered.scatter_(2, nn_idx[:, :, :, None].repeat(1, 1, 1, h_message.shape[3]), h_message)
        dh = torch.div(torch.sum(h_message_scattered, dim=1), 
                       Y_scale[:,:,None].repeat(1,1,h_message_scattered.shape[-1])) # [B,l,C]
        dh = gather_context_atom_features(dh, nn_idx) # [B,L,M,C]

        # update neighborhood ligand nodes
        h_Y_nodes = self.norm1(h_Y_nodes + self.dropout1(dh))
        dh = self.dense(h_Y_nodes)
        h_Y_nodes = self.norm2(h_Y_nodes + self.dropout2(dh))
        if mask_V is not None:
            h_Y_nodes = mask_V.unsqueeze(-1).unsqueeze(-1) * h_Y_nodes

        #  central residue nodes, neighborhood ligand nodes --> update protein-ligand graph edges
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_Y_nodes.size(-2), -1)
        h_YEV = torch.cat([h_Y_nodes, h_E_context, h_V_expand], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_YEV))))) # [B,L,M,C]
        h_E_context = self.norm3(h_E_context + self.dropout3(h_message))
        return h_Y_nodes, h_E_context


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


side_chain_atom_types = [
    6,
    6,
    6,
    8,
    8,
    16,
    6,
    6,
    6,
    7,
    7,
    8,
    8,
    16,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    8,
    8,
    6,
    7,
    7,
    8,
    6,
    6,
    6,
    7,
    8,
]

periodic_table_features = [
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
    ],
    [
        0,
        1,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ],
    [
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ],
]
