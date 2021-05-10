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
    # edge_index = torch.tensor(edge_index, dtype=torch.long).unsqueeze(-1)
    return edge_index


class NTIGADataset_simulator(PyGdata.Dataset):
    def __init__(
        self, path, root=None, transform=None, pre_transform=None, pre_filter=None
    ):
        p = os.path.dirname(path)
        self.file_path = path

        self.graph_edge_file = p + "/pipe_graph_topo_17_local.txt"

        self.edge_index = edge_index_from_file(self.graph_edge_file)

        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file["feature"])
            self.feature = file["feature"][:]
            self.target = file["target"][:]

        self.num_in = self.feature.shape[-1]
        self.num_out = self.target.shape[-1]

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        # X = torch.from_numpy(self.feature[idx, :, :, :]).unsqueeze(0)
        # Y = torch.from_numpy(self.target[idx, :, :, :]).unsqueeze(0)
        # graph_data = Data(x=X, edge_index=self.edge_index.unsqueeze(-1), y=Y)

        X = torch.from_numpy(self.feature[idx, :, :, :]).permute(1, 2, 0)
        Y = torch.from_numpy(self.target[idx, :, :, :]).permute(1, 2, 0)
        graph_data = Data(x=X, edge_index=self.edge_index, y=Y)
        return graph_data


class NTIGADataset_new(PyGdata.Dataset):
    def __init__(
        self,
        file_path,
        recursive,
        load_data,
        data_cache_size=3,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        # super(NTIGADataset, self).__init__(root, transform, pre_transform)

        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        p = Path(file_path)
        assert p.is_dir()
        if recursive:
            files = sorted(p.glob("**/*.h5"))
        else:
            files = sorted(p.glob("*.h5"))
        if len(files) < 1:
            raise RuntimeError("No hdf5 datasets found")

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

        self.graph_edge_file = file_path + "/pipe_graph_topo_17_local.txt"
        self.edge_index = edge_index_from_file(self.graph_edge_file)

        #         p = os.path.dirname(path)
        # self.file_path = path
        # self.feature = None
        # self.target = None
        # with h5py.File(self.file_path, "r") as file:
        #     self.dataset_len = len(file["feature"])

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        return

    def __getitem__(self, index):
        X = self.get_data("feature", index)
        Y = self.get_data("target", index)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        graph_data = Data(x=X, edge_index=self.edge_index, y=Y)

        return graph_data

    def __len__(self):
        return len(self.get_data_infos("feature"))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, "r") as h5_file:
            # Walk through all groups, extracting datasets
            # for gname, group in h5_file.items():
            # for dname, ds in group.items():
            for dname, ds in h5_file.items():
                for _tensor in ds[()]:
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(_tensor, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {
                            "file_path": file_path,
                            "type": dname,
                            "shape": _tensor.shape,
                            "cache_idx": idx,
                        }
                    )

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, "r", swmr=True) as h5_file:
            # for gname, group in h5_file.items():
            # for dname, ds in group.items():
            for dname, ds in h5_file.items():
                for _tensor in ds[()]:
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(_tensor, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(
                        i
                        for i, v in enumerate(self.data_info)
                        if v["file_path"] == file_path
                    )

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]["cache_idx"] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {
                    "file_path": di["file_path"],
                    "type": di["type"],
                    "shape": di["shape"],
                    "cache_idx": -1,
                }
                if di["file_path"] == removal_keys[0]
                else di
                for di in self.data_info
            ]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di["type"] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]["file_path"]
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]["cache_idx"]
        return self.data_cache[fp][cache_idx]
