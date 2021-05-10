import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.data as PyGdata

import os
import glob
import pandas as pd
import numpy as np
import h5py
import re
import sys
from pathlib import Path


def edge_index_from_file(graph_file):
    edge_index = np.genfromtxt(graph_file, delimiter="\t", dtype=int)
    edge_index = np.transpose(edge_index)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index


class NTIGADataset_tree(PyGdata.Dataset):
    def __init__(
        self, path, root=None, transform=None, pre_transform=None, pre_filter=None
    ):
        p = os.path.dirname(path)
        self.file_path = path
        # tree level
        self.tree_info = {}
        self.tree_sim_dset = {}
        self.tree_sim_info = {}

        with h5py.File(self.file_path, "r") as file:
            for tree_name, tree_data in file.items():
                print(type(tree_name))
                self.tree_info[tree_name] = file[tree_name].attrs["graph_rep"]
                sim_info = {}
                sim_dset = {}

                for sim_name, sim_data in tree_data.items():
                    sim_info[sim_name] = {
                        "type": tree_data[sim_name].attrs["type"],
                        "edge": tree_data[sim_name].attrs["edge"],
                    }
                    sim_dset[sim_name] = {
                        "feature": sim_data["feature"][:],
                        "target": sim_data["target"][:],
                    }
                self.tree_sim_info[tree_name] = sim_info
                self.tree_sim_dset[tree_name] = sim_dset

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return

    def __len__(self):
        return len(self.tree_info)

    def __getitem__(self, idx):
        tree_name = "tree_{}".format(idx + 1)
        return (
            self.tree_info[tree_name],
            self.tree_sim_info[tree_name],
            self.tree_sim_dset[tree_name],
        )
