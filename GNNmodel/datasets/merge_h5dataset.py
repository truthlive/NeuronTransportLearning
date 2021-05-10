import os
import glob
import pandas as pd
import numpy as np
import h5py
import re
import sys
from pathlib import Path


work_path = "/pylon5/eg5fp3p/rachels/angranl_stuff/data/Pipe/"
data_path = work_path + "PipeDataAll_17_time/"

p = Path(data_path)
assert p.is_dir()

# new_data = h5py.File(data_path + "PipeDataAll_merge.hdf5", "a")
num_bc = 200
num_tstep = 100
num_layer = 8
num_node_per_section = 17
geo_perdataset = 10


files = sorted(p.glob("*.h5"))
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")

num_geo = len(files) * geo_perdataset
feature_shape = (num_bc * num_tstep * num_geo, num_layer * num_node_per_section, 5)
target_shape = (num_bc * num_tstep * num_geo, num_layer * num_node_per_section, 1)

with h5py.File(data_path + "PipeDataAll_merge.h5", "a") as new_data:
    new_data.create_dataset("feature", feature_shape, np.float32)
    new_data.create_dataset("target", target_shape, np.float32)
    for i, h5dataset_fp in enumerate(files):
        data = h5py.File(h5dataset_fp, "r")
        idx_start = i * num_bc * num_tstep * geo_perdataset
        idx_end = (i + 1) * num_bc * num_tstep * geo_perdataset
        new_data["feature"][idx_start:idx_end, :, :] = data["feature"][:, :, :]
        new_data["target"][idx_start:idx_end, :, :] = data["target"][:, :, :]
