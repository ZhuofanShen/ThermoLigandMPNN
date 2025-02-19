import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from datasets import MegaScaleDataset, FireProtDataset, MdrDataset, ddgBenchDataset
from thermompnn_train import TransferModelPL


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:

    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1
    # Compute centralities
    # centralities = {
    #     'all': torch.ones(num_neighbors.shape, device=num_neighbors.device),
    #     'core': num_neighbors >= core_threshold,
    #     # 'boundary': num_neighbors < core_threshold & num_neighbors > surface_threshold,
    #     'surface': num_neighbors <= surface_threshold,
    # }
    return num_neighbors

def get_metrics():
    return {
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson":  PearsonCorrCoef().to('cuda'),
    }

def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics(),
    }
    print('Testing Model %s on dataset %s' % (name, dataset_name))

    for i, batch in enumerate(tqdm(dataset)):
        pdb, mutations = batch
        pred, _ = model(pdb, mutations)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    raw_pred_df = pd.DataFrame(
        columns=['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
                 'neighbors', 'best_AA'])
    metrics = {
        "ddG": get_metrics(),
    }
    print('Running model %s on dataset %s' % (name, dataset_name))
    for i, batch in enumerate(tqdm(dataset)):
        mut_pdb, mutations = batch
        pred, _ = model(mut_pdb, mutations)

        if centrality:
            coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain,
                                           radius=10.)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

                # assign raw preds and useful details to df
                col_list = ['ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                val_list = [mut.ddG.cpu().item(), out["ddG"].cpu().item(), mut.position, mut.wildtype,
                            mut.mutation, mut.pdb.strip('.pdb')]
                for col, val in zip(col_list, val_list):
                    raw_pred_df.loc[row, col] = val

                if centrality:
                    raw_pred_df.loc[row, 'neighbors'] = neighbors[mut.position].cpu().item()

            raw_pred_df.loc[row, 'Model'] = name
            raw_pred_df.loc[row, 'Dataset'] = dataset_name
            # if 'Megascale' not in dataset_name: # different pdb column formatting
            #     key = mut.pdb
            # else:
            #     key = mut.pdb + '.pdb'
            # if 'S669' not in dataset_name: # S669 is missing WT seq info - omit to prevent error
            #     raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        if max_batches is not None and i >= max_batches:
            break
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:  # , "dTm"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                pass
    results.append(column)
    raw_pred_df.to_csv(name + '_' + dataset_name + "_raw_preds.csv")
    del raw_pred_df

    return results


def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            # 'num_workers': 8,
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

    models = {
        "ThermoMPNN": TransferModelPL.load_from_checkpoint(cfg.platform.model_path, cfg=cfg).model
    }

    # misc_data_loc = '/scratch/zs251/ThermoMPNN/data'
    datasets = {
        # "MdrDB": MdrDataset("/scratch/zs251/Thermo-Ligand-MPNN/data/MdrDB_CoreSet_v1.0.2022/MdrDB_CoreSet_release_v1.0.2022.old.tsv", "/scratch/zs251/Thermo-Ligand-MPNN/data/MdrDB_CoreSet_v1.0.2022/Single_Substitution/MdrDB_CoreSet_single_substitution.old.pkl", "all", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),

        # "MdrDB_AIMMS": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "AIMMS.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["AIMMS"]),
        # "MdrDB_Depmap": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "Depmap.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["Depmap"]),
        # "MdrDB_GDSC": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "GDSC.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["GDSC"]),
        # "MdrDB_KinaseMD": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "KinaseMD.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["KinaseMD"]),
        # "MdrDB_Platinum": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "Platinum.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["Platinum"]),
        # "MdrDB_RET": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "RET.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["RET"]),
        # "MdrDB_TKI": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "TKI.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["TKI"]),

        # "MdrDB_AIMMS": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "AIMMS.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_Depmap": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "Depmap.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_GDSC": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "GDSC.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_KinaseMD": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "KinaseMD.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_Platinum": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "Platinum.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_RET": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "RET.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        # "MdrDB_TKI": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits[:-3] + "TKI.pkl", "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),

        # "MdrDB": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits, "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["all"]),
        "MdrDB": MdrDataset(cfg.data_loc.mdrdb_tsv, cfg.data_loc.mdrdb_splits, "test", cfg.data_loc.mdrdb_pdbs, ["Single Substitution"], ["AIMMS", "Platinum", "TKI"]),

        # "Megascale-test": MegaScaleDataset(cfg, "test"),
        # "Fireprot-test": FireProtDataset(cfg, "test"),
        # "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free"),

        # "P53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv')),
        # "MYOGLOBIN": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')),
        # "SSYM_dir": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')),
        # "SSYM_inv": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')),
        # "S669": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'S669/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'S669/s669_clean_dir.csv'))
    }

    results = []

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                results = run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=args.centrality)
            else:
                results = run_prediction_default(name, model, dataset_name, dataset, results)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("ThermoMPNN_metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_preds', action='store_true', default=False, help='Save raw model predictions as csv')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')

    args = parser.parse_args()
    cfg = OmegaConf.load("local.yaml")
    with torch.no_grad():
        main(cfg, args)
