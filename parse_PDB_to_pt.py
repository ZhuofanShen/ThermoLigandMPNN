import argparse
import os
import pickle
from protein_mpnn_utils import parse_PDB


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()

for pdb in filter(lambda x: x.endswith(".pdb"), os.listdir(args.path)):
    pdb_path = os.path.join(args.path, pdb)
    pdb_dict = parse_PDB(pdb_path)
    pickle.dump(pdb_dict, open(pdb_path[:-3] + "pkl", "wb"))
