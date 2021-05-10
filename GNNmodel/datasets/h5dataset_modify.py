import os
import glob
import pandas as pd
import numpy as np
import h5py
import re
import sys
from pathlib import Path

fname = "/pylon5/eg5fp3p/rachels/angranl_stuff/data/Pipe/PipeDataAll_17_test/PipeDataSmall_17.h5"
fname_new = "/pylon5/eg5fp3p/rachels/angranl_stuff/data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_ghost.h5"

num_bc = 50
num_tstep = 100
num_layer = 8
num_node_per_section = 17
num_geo = 1

old_data = h5py.File(fname)
feature_shape = (
    num_bc * num_tstep * num_geo,
    (num_layer + 2) * num_node_per_section,
    5,
)
target_shape = (num_bc * num_tstep * num_geo, (num_layer + 2) * num_node_per_section, 1)

with h5py.File(fname_new) as new_data:
    new_data.create_dataset("feature", feature_shape, np.float32)
    new_data.create_dataset("target", target_shape, np.float32)
    new_data["feature"][:, 0 : (num_layer) * num_node_per_section, :] = old_data[
        "feature"
    ]
    new_data["feature"][
        :,
        (num_layer) * num_node_per_section : (num_layer + 1) * num_node_per_section,
        :,
    ] = old_data["feature"][:, 0:num_node_per_section, :]

    new_data["feature"][
        :,
        (num_layer + 1) * num_node_per_section : (num_layer + 2) * num_node_per_section,
        :,
    ] = old_data["feature"][
        :,
        (num_layer - 1) * num_node_per_section : (num_layer) * num_node_per_section,
        :,
    ]

    new_data["target"][:, 0 : (num_layer) * num_node_per_section, :] = old_data[
        "target"
    ]
    new_data["target"][
        :,
        (num_layer) * num_node_per_section : (num_layer + 1) * num_node_per_section,
        :,
    ] = old_data["target"][:, 0:num_node_per_section, :]

    new_data["target"][
        :,
        (num_layer + 1) * num_node_per_section : (num_layer + 2) * num_node_per_section,
        :,
    ] = old_data["target"][
        :,
        (num_layer - 1) * num_node_per_section : (num_layer) * num_node_per_section,
        :,
    ]

old_data.close()
