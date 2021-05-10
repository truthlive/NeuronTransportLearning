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
    "--geo-start", type=int, default=0, help="the beginning index of the geometry"
)
parser.add_argument(
    "--h5-index", type=int, default=0, help="the beginning index of the h5 file"
)
parser.add_argument(
    "--num-geo", type=int, default=10, help="the number of output geometries"
)
parser.add_argument(
    "--num-bc",
    type=int,
    default=1,
    help="the number of output boundary condition settings",
)
parser.add_argument(
    "--num-tstep", type=int, default=100, help="the number of time steps",
)
parser.add_argument(
    "--data-path",
    type=str,
    default=None,
    metavar="path",
    help="path to the raw dataset",
)
parser.add_argument(
    "--out-fname",
    type=str,
    default=None,
    metavar="fname",
    help="output filename of the processed dataset",
)
parser.add_argument(
    "--mode",
    type=str,
    default=None,
    metavar="fname",
    help="output filename of the processed dataset",
)

# parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default= 100, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
# parser.add_argument('--dropout_rate', type=float, default=0.0, help="dropout ratio")
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
# parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
args = parser.parse_args()

num_tree = args.num_tree
num_bc = args.num_bc
num_tstep = args.num_tstep
geo_start = args.geo_start
h5_index = args.h5_index

print("num_tree: ", num_tree)
print("geo_start: ", geo_start)
print("h5_index: ", h5_index)


num_tstep = 100
num_layer = 8
num_node_per_section = 17
workdir = args.data_path

# * Input path setting
fname_meshinfo = workdir + "controlmesh_info.txt"
fname_graphrep = workdir + "GraphRep.swc"
fname_C2Dmapping = workdir + "C2Dmapping_extract.txt"


# * Output path setting
path_out = args.data_path + "processed/"
if not os.path.exists(path_out):
    os.makedirs(path_out)
fname_h5 = path_out + args.out_fname
print(fname_h5)
# fp_h5 = h5py.File(fname_h5, "w", rdcc_nbytes=1024 ** 2 * 200)


# * Input info for simulator
num_pt, num_ele = np.loadtxt(fname_meshinfo, dtype=int)
graph_rep, num_sim = process.ReadGraphRep(fname_graphrep)

grpname_tree = "tree_{}".format(1)

if args.mode == "sim":
    # ! Output h5 data structure: tree grp -> simulator subgrp
    with h5py.File(fname_h5, "w", rdcc_nbytes=1024 ** 2 * 200) as fp_h5:
        grp_tree = fp_h5.create_group(grpname_tree)
        grp_tree.attrs["graph_rep"] = graph_rep
        List_line_xyz = []
        List_line_vals = []
        List_num_node = []

        # ! Setup data structure: define tree and simulator group; define dataset size
        for idx_sim in range(num_sim):
            subgrpname_sim = "sim_{}".format(idx_sim + 1)
            subgrp_sim = grp_tree.create_group(subgrpname_sim)

            fname_edge = workdir + "Simulator/simulator_{}_edge.txt".format(idx_sim + 1)
            fname_S2Dmapping = workdir + "Simulator/simulator_{}_S2Dmapping.txt".format(
                idx_sim + 1
            )
            S2Dmapping = process.ReadS2DMapping(fname_S2Dmapping)
            edge = process.ReadSimulatorEdge(fname_edge)
            line_xyz, line_vals = process.ExtractLineList(S2Dmapping, num_pt, num_ele)
            num_node = S2Dmapping.shape[0]

            List_line_xyz.append(line_xyz)
            List_line_vals.append(line_vals)
            List_num_node.append(num_node)

            feature_shape = (
                num_bc * num_tstep,
                num_node,
                5,
            )
            target_shape = (
                num_bc * num_tstep,
                num_node,
                1,
            )

            subgrp_sim.attrs["type"] = graph_rep[idx_sim][1]
            subgrp_sim.attrs["edge"] = edge
            subgrp_sim.create_dataset("feature", feature_shape)
            subgrp_sim.create_dataset("target", target_shape)

        for idx_sim in range(num_sim):
            subgrpname_sim = "sim_{}".format(idx_sim + 1)
            num_node = List_num_node[idx_sim]
            for j in range(0, num_bc):
                fname_sim_para = workdir + "simulation_parameter/{bc:0>4d}.txt".format(
                    bc=j + 1
                )
                sim_para = process.ReadSimulationParameter(fname_sim_para)

                vec_bc1 = np.ones((num_node_per_section, 1), dtype=np.float32)
                vec_bc2 = np.zeros(
                    (num_node - num_node_per_section, 1), dtype=np.float32
                )
                vec_bc1 = vec_bc1 * (
                    sim_para["N0bc"] + sim_para["Nplusbc"] + sim_para["Nminusbc"]
                )
                vec_bc_root = np.concatenate((vec_bc1, vec_bc2), axis=0)
                vec_bc_nonroot = np.zeros((num_node, 1), dtype=np.float32)

                t0_s = time.time()
                for k in range(0, num_tstep):
                    fname_IGA = (
                        workdir
                        + "{bc:0>4d}/controlmesh_allparticle_{t}.vtk".format(
                            bc=j + 1, t=k + 1
                        )
                    )

                    t0_k = time.time()

                    if j == 0:
                        pts, vals = process.ReadAndExtractVTK_List(
                            fname_IGA,
                            List_line_xyz[idx_sim],
                            List_line_vals[idx_sim],
                            j,
                        )
                    else:
                        _, vals = process.ReadAndExtractVTK_List(
                            fname_IGA,
                            List_line_xyz[idx_sim],
                            List_line_vals[idx_sim],
                            j,
                        )
                    t1_k = time.time()

                    # print(
                    #     "Data extraction {:d} Done! Time: {:.4f}".format(
                    #         idx_sim, t1_k - t0_k
                    #     )
                    # )
                    vec_t = np.ones((num_node, 1), dtype=np.float32)
                    vec_t = vec_t * (k + 1) * 0.1

                    mat_input = np.array(pts)
                    if graph_rep[idx_sim][-1] == -1:
                        mat_input = np.concatenate((pts, vec_t, vec_bc_root), axis=1)
                    else:
                        mat_input = np.concatenate((pts, vec_t, vec_bc_nonroot), axis=1)

                    mat_output = np.array(vals)

                    grp_tree[subgrpname_sim]["feature"][
                        k + j * num_tstep, ...
                    ] = mat_input
                    grp_tree[subgrpname_sim]["target"][
                        k + j * num_tstep, ...
                    ] = mat_output

                # print("Sample {:d} Done! Time: {:.4f}".format(k, t1_s - t0_s))
            print("Simulator {:d} Done!".format(idx_sim))

elif args.mode == "tree":
    # ! Output h5 data structure: entire tree
    with h5py.File(fname_h5, "w", rdcc_nbytes=1024 ** 2 * 200) as fp_h5:
        for i in range(num_tree):
            
            C2Dmapping = process.ReadC2DMapping(fname_C2Dmapping)
            line_xyz, line_vals = process.ExtractLineList(
                C2Dmapping, num_pt, num_ele
            )
            num_node = C2Dmapping.shape[0]

            feature_shape = (
                num_bc * num_tstep,
                num_node,
                5,
            )
            target_shape = (
                num_bc * num_tstep,
                num_node,
                1,
            )

            subgrp_sim.attrs["type"] = graph_rep[idx_sim][1]
            subgrp_sim.attrs["edge"] = edge
            subgrp_sim.create_dataset("feature", feature_shape)
            subgrp_sim.create_dataset("target", target_shape)


            for j in range(0, num_bc):

                fname_IGA_BC = (
                    workdir + "{bc:0>4d}/controlmesh_allparticle_0.vtk".format(bc=j + 1)
                )
                _, vals = process.ReadAndExtractVTK_List(
                    fname_IGA, line_xyz, line_vals, j,
                )
                vec_bc = vals

                t0_s = time.time()
                for k in range(0, num_tstep):
                    fname_IGA = (
                        workdir
                        + "{bc:0>4d}/controlmesh_allparticle_{t}.vtk".format(
                            bc=j + 1, t=k + 1
                        )
                    )

                    t0_k = time.time()

                    if j == 0:
                        pts, vals = process.ReadAndExtractVTK_List(
                            fname_IGA, line_xyz, line_vals, j,
                        )
                    else:
                        pts, vals = process.ReadAndExtractVTK_List(
                            fname_IGA, line_xyz, line_vals, j,
                        )
                    t1_k = time.time()

                    # print(
                    #     "Data extraction {:d} Done! Time: {:.4f}".format(
                    #         idx_sim, t1_k - t0_k
                    #     )
                    # )
                    vec_t = np.ones((num_node, 1), dtype=np.float32)
                    vec_t = vec_t * (k + 1) * 0.1

                    mat_input = np.array(pts)
                    if graph_rep[idx_sim][-1] == -1:
                        mat_input = np.concatenate((pts, vec_t, vec_bc_root), axis=1)
                    else:
                        mat_input = np.concatenate((pts, vec_t, vec_bc_nonroot), axis=1)

                    mat_output = np.array(vals)

                    grp_tree[subgrpname_sim]["feature"][
                        k + j * num_tstep, ...
                    ] = mat_input
                    grp_tree[subgrpname_sim]["target"][
                        k + j * num_tstep, ...
                    ] = mat_output

                # print("Sample {:d} Done! Time: {:.4f}".format(k, t1_s - t0_s))
            print("Simulator {:d} Done!".format(idx_sim))
