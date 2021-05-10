#%%
import torch
import numpy as np
import h5py

from torch.utils import data
# from torch_geometric.data import Data

#%%
def ReadAndExtractVTK(filename, flag_xyz, node_extract_list):
    pt_count = 0
    n_pts = -1
    num_extract = len(node_extract_list)
    xyz_start = -1
    xyz_end = -1
    xyz_it = 0
    val_start = -1
    val_end = -1
    val_it = 0
    tmp_mat = []
    pts = []
    vals = []
    with open(filename) as fp:
        for count, line in enumerate(fp):
            if line.strip():
                tmp_old = line.split(" ")
                tmp = [elem for elem in tmp_old if elem.strip()]

                if tmp[0] == "POINTS":
                    n_pts = int(tmp[1])
                    xyz_start = count + 1
                    xyz_end = xyz_start + n_pts

                if tmp[0] == "POINT_DATA":
                    n_pts = int(tmp[1])
                    val_start = count + 3
                    val_end = val_start + n_pts

                if (
                    count >= xyz_start
                    and count < xyz_end
                    and flag_xyz == 0
                    and xyz_it < num_extract
                ):
                    if count - xyz_start == node_extract_list[xyz_it]:
                        tmp_xyz = np.fromstring(line, dtype=float, sep=" ")
                        pts.append(tmp_xyz.tolist())
                        xyz_it += 1

                if count >= val_start and count < val_end and val_it < num_extract:
                    if count - val_start == node_extract_list[val_it]:
                        tmp_vals = np.fromstring(line, dtype=float, sep=" ")
                        vals.append(tmp_vals.tolist())
                        val_it += 1
                
            if count >= val_end and val_end > 0:
                fp.close()
                return pts, vals

    return pts, vals


# fname = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/0001/0001/controlmesh_allparticle_0.vtk"
# flag_xyz = 1
# node_extract_list = [0, 4, 8]

# pts, vals = ReadAndExtractVTK(fname, flag_xyz, node_extract_list)


# def ReadVTK_CM(filename):
#     pt_count = 0
#     ele_count = 0
#     n_ele = -1
#     ele_start = -1
#     ele_end = -1
#     tmp_mat = []
#     pts = []
#     tmesh = []
#     with open(filename) as fp:
#         for count, line in enumerate(fp):
#             if line.strip():
#                 tmp_old = line.split(" ")
#                 tmp = [elem for elem in tmp_old if elem.strip()]

#                 if tmp[0] == "CELLS":
#                     n_ele = int(tmp[1])
#                     ele_start = count + 1
#                     ele_end = ele_start + n_ele

#                 if count >= ele_start and count < ele_end:
#                     tmp_cnct = np.fromstring(line, dtype=int, sep=" ")
#                     tmp_cnct = tmp_cnct[1:]
#                     tmp_cmele = ControlElement(tmp_cnct)
#                     tmesh.append(tmp_cmele)
#     return pts, tmesh


# %%
