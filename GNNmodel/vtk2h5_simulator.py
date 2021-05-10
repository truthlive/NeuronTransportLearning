#%%
import os
import h5py
import numpy as np
import os.path
from os import path
import sys
import argparse
import time
import utils.data_process as process

parser = argparse.ArgumentParser(description="Postprocessor of pipe")
# parser.add_argument('--path', type=os.path.abspath, required=True, help="data path")
# parser.add_argument('--test_ratio', type=float, required=True, help="percentage of test data")
parser.add_argument(
    "--geo_start", type=int, default=0, help="the beginning index of the geometry"
)
parser.add_argument(
    "--h5_index", type=int, default=0, help="the beginning index of the h5 file"
)
parser.add_argument(
    "--num_geo", type=int, default=10, help="the number of output geometries"
)
parser.add_argument(
    "--num_bc",
    type=int,
    default=200,
    help="the number of output boundary condition settings",
)
parser.add_argument(
    "--data-path", type=str, default=None, metavar="path", help="path to the dataset",
)

# parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default= 100, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
# parser.add_argument('--dropout_rate', type=float, default=0.0, help="dropout ratio")
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
# parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
opt = parser.parse_args()

num_geo = opt.num_geo
num_bc = opt.num_bc
geo_start = opt.geo_start
h5_index = opt.h5_index

print("num_geo: ", num_geo)
print("geo_start: ", geo_start)
print("h5_index: ", h5_index)


num_tstep = 100
num_layer = 8
num_node_per_section = 17
workdir = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/"
workdir_h5 = workdir + "HDF5/"
fname_h5 = workdir_h5 + "PipeDataAll_17_xyz_{}.h5".format(h5_index)
# fname_h5 = workdir_h5 + "PipeDataAll_17_rstheta_{}.h5".format(h5_index)

fname_edge = workdir + "pipe_graph_topo_17_local.txt"
fname_meshinfo = workdir + "controlmesh_info.txt"
fname_S2Dmapping = workdir + "S2Dmapping.txt"

num_pt, num_ele = np.loadtxt(fname_meshinfo, dtype=int)

S2Dmapping = process.ReadS2DMapping(fname_S2Dmapping)
line_xyz, line_vals = process.ExtractLineList(S2Dmapping, num_pt, num_ele)
edge = process.ReadSimulatorEdge(fname_edge)

edge_shape = (num_bc * num_tstep * num_geo, edge.shape[0], edge.shape[1])
feature_shape = (num_bc * num_tstep * num_geo, num_layer * num_node_per_section, 5)
target_shape = (num_bc * num_tstep * num_geo, num_layer * num_node_per_section, 1)

node_list = []
theta_list = []
bc_list = []

for i in range(num_layer):
    theta_list.append(0.0 / 180.0 * np.pi)
    theta_list.append(180.0 / 180.0 * np.pi)
    theta_list.append(90.0 / 180.0 * np.pi)
    theta_list.append(0.0 / 180.0 * np.pi)
    theta_list.append(180.0 / 180.0 * np.pi)
    theta_list.append(90.0 / 180.0 * np.pi)
    theta_list.append(135.0 / 180.0 * np.pi)
    theta_list.append(0.0 / 180.0 * np.pi)
    theta_list.append(45.0 / 180.0 * np.pi)
    theta_list.append(135.0 / 180.0 * np.pi)
    theta_list.append(45.0 / 180.0 * np.pi)
    theta_list.append(270.0 / 180.0 * np.pi)
    theta_list.append(270.0 / 180.0 * np.pi)
    theta_list.append(225.0 / 180.0 * np.pi)
    theta_list.append(315.0 / 180.0 * np.pi)
    theta_list.append(225.0 / 180.0 * np.pi)
    theta_list.append(315.0 / 180.0 * np.pi)

for i in range(0, num_layer):
    # node_list.append(0 + 201 * i)
    # node_list.append(1 + 201 * i)
    # node_list.append(2 + 201 * i)
    # node_list.append(3 + 201 * i)
    # node_list.append(7 + 201 * i)
    # node_list.append(13 + 201 * i)
    # node_list.append(20 + 201 * i)
    # node_list.append(27 + 201 * i)
    # node_list.append(35 + 201 * i)
    # node_list.append(62 + 201 * i)
    # node_list.append(96 + 201 * i)
    # node_list.append(108 + 201 * i)
    # node_list.append(112 + 201 * i)
    # node_list.append(119 + 201 * i)
    # node_list.append(128 + 201 * i)
    # node_list.append(155 + 201 * i)
    # node_list.append(189 + 201 * i)
    for j in range(0, 17):
        if i == 0:
            bc_list.append([1, 2])
        else:
            bc_list.append([-1, -1])


#%%

with h5py.File(fname_h5, "w", rdcc_nbytes=1024 ** 2 * 200) as f:
    # with h5py.File(fname_h5,'w') as f:
    f.create_dataset("feature", feature_shape, np.float32)
    f.create_dataset("target", target_shape, np.float32)
    for i in range(geo_start, geo_start + num_geo):

        # f = h5py.File(workdir_h5+'Geo{:0>4d}.hdf5'.format(i),'w')
        # grp = f.create_group('/Geo{:0>4d}'.format(i + 1))
        # grp.create_dataset("feature", feature_shape, np.float32)
        # grp.create_dataset("target", target_shape, np.float32)
        t0 = time.time()
        for j in range(0, num_bc):
            vec_bc1 = np.ones((num_node_per_section, 1), dtype=np.float32)
            vec_bc2 = np.zeros(
                ((num_layer - 1) * num_node_per_section, 1), dtype=np.float32
            )
            vec_bc1 = vec_bc1 * (j + 1) * 0.05
            vec_bc = np.concatenate((vec_bc1, vec_bc2), axis=0)
            t0_s = time.time()
            for k in range(0, num_tstep):
                fname = (
                    workdir
                    + "{geo:0>4d}/{bc:0>4d}/controlmesh_allparticle_{t}.vtk".format(
                        geo=i + 1, bc=j + 1, t=k + 1
                    )
                )

                t0_k = time.time()
                # if j == 0:
                #     pts, vals = process.ReadAndExtractVTK(fname, node_list, j)
                # else:
                #     _, vals = process.ReadAndExtractVTK(fname, node_list, j)

                if j == 0:
                    pts, vals = process.ReadAndExtractVTK_List(
                        fname, line_xyz, line_vals, j
                    )
                    # pts = process.EncodePipeGeo(
                    #     pts, theta_list, num_layer, num_node_per_section
                    # )
                else:
                    _, vals = process.ReadAndExtractVTK_List(
                        fname, line_xyz, line_vals, j
                    )
                t1_k = time.time()
                print("Data extraction {:d} Done! Time: {:.4f}".format(j, t1_k - t0_k))
                vec_t = np.ones((num_layer * num_node_per_section, 1), dtype=np.float32)
                vec_t = vec_t * (k + 1) * 0.1

                mat_input = np.array(pts)
                mat_input = np.concatenate((pts, vec_t, vec_bc), axis=1)

                mat_output = np.array(vals)

                # print(mat_output.shape)

                # grp["feature"][k+j*num_tstep,...] = mat_input
                # grp["target"][k+j*num_tstep,...] = mat_output

                f["feature"][
                    k + j * num_tstep + (i - geo_start) * num_tstep * num_bc, ...
                ] = mat_input
                f["target"][
                    k + j * num_tstep + (i - geo_start) * num_tstep * num_bc, ...
                ] = mat_output

                # print("Sample {:d} Done! Time: {:.4f}".format(k, t1_s - t0_s))
            t1_s = time.time()
            # print("BC group {:d} Done! Time: {:.4f}".format(j, t1_s - t0_s))
        t1 = time.time()
        print("Group {:d} Done! Time: {:.4f}".format(i + 1, t1 - t0))

print("Done!")


# %%
# import h5py
# workdir = '/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/'
# workdir_h5 = workdir + 'HDF5/'
# fname_h5 = workdir_h5 + 'PipeDataAll_17_{}.h5'.format(2)
# with h5py.File(fname_h5,'r') as f_debug:
#     for gname in f_debug.keys():
#         print(f_debug[gname].keys())
#         print(f_debug[gname]['feature'][3,:,:])
# %%
