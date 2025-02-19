import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from Bio.PDB import PDBParser

from datasets import Mutation
from thermompnn_train import TransferModelPL
from protein_mpnn_utils import parse_PDB


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def get_ssm_mutations(pdb):
    # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)
    return mutation_list

def get_chains(pdb):
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb)
  chains = [c.id for c in structure.get_chains()]
  return chains


def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            # 'num_workers': 2,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        },
        'proteinmpnn_model': {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_letters': 21,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    # load the chosen model and dataset
    models = {"ThermoMPNN": TransferModelPL.load_from_checkpoint(cfg.platform.model_path, cfg=cfg).model}

    input_pdb = args.pdb
    pdb_id = os.path.basename(input_pdb).rstrip('.pdb')

    datasets = {
        pdb_id: args.pdb
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation',])
    row = 0
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if len(args.chain) < 1:  # if unspecified, take first chain
                chain = get_chains(input_pdb)[0]
            else:
                chain = args.chain
            mut_pdb = parse_PDB(input_pdb, chains=list(chain))
            mutation_list = get_ssm_mutations(mut_pdb[0])
            final_mutation_list = []

            # build into list of Mutation objects
            for n, m in enumerate(mutation_list):
                if m is None:
                    final_mutation_list.append(None)
                    continue
                m = m.strip()  # clear whitespace
                wtAA, position, mutAA = str(m[0]), int(str(m[1:-1])), str(m[-1])

                assert wtAA in ALPHABET, f"Wild type residue {wtAA} invalid, please try again with one of the following options: {ALPHABET}"
                assert mutAA in ALPHABET, f"Wild type residue {mutAA} invalid, please try again with one of the following options: {ALPHABET}"
                mutation_obj = Mutation(position=position, wildtype=wtAA, mutation=mutAA,
                                        ddG=None, pdb=mut_pdb[0]['name'])
                final_mutation_list.append(mutation_obj)

            pred, _ = model(mut_pdb, final_mutation_list)

            for mut, out in zip(final_mutation_list, pred):
                if mut is not None:
                    col_list = ['ddG_pred', 'position', 'wildtype', 'mutation', 'pdb', 'chain']
                    val_list = [out["ddG"].cpu().item(), mut.position, mut.wildtype,
                                mut.mutation, mut.pdb.strip('.pdb'), chain]
                    for col, val in zip(col_list, val_list):
                        raw_pred_df.loc[row, col] = val

                    raw_pred_df.loc[row, 'Model'] = name
                    raw_pred_df.loc[row, 'Dataset'] = dataset_name
                    row += 1

    print(raw_pred_df)
    raw_pred_df.to_csv("ThermoMPNN_inference_%s.csv" % pdb_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='', help='Input PDB to use for custom inference')
    parser.add_argument('--chain', type=str, default='A', help='Chain in input PDB to use.')

    args = parser.parse_args()
    cfg = OmegaConf.load("local.yaml")
    with torch.no_grad():
        main(cfg, args)
