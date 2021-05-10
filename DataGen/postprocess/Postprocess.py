#%%
import os
import utils
import h5py
import numpy as np
import os.path
from os import path 
import sys
import argparse

parser = argparse.ArgumentParser(description='Postprocessor of pipe')
# parser.add_argument('--path', type=os.path.abspath, required=True, help="data path")
# parser.add_argument('--test_ratio', type=float, required=True, help="percentage of test data")
parser.add_argument('--geo_start', type=int, default=0, help='the beginning index of the geometry')
parser.add_argument('--h5_index', type=int, default=0, help='the beginning index of the h5 file')
parser.add_argument('--num_geo', type=int, default=10, help='the number of output geometries')

# parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default= 100, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
# parser.add_argument('--dropout_rate', type=float, default=0.0, help="dropout ratio")
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
# parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
opt = parser.parse_args()

num_geo = opt.num_geo
geo_start = opt.geo_start
h5_index = opt.h5_index

print(num_geo)
print(geo_start)
print(h5_index)

num_bc = 200
num_tstep = 100
num_layer = 8
num_node_per_section = 17
workdir = '/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/'
workdir_h5 = workdir + 'HDF5/'
fname_h5 = workdir_h5 + 'PipeDataAll_17_{}.h5'.format(h5_index)
fname_check = workdir_h5 + 'data_check.txt'

feature_shape = (num_bc*num_tstep, num_layer * num_node_per_section, 5)
target_shape = (num_bc*num_tstep, num_layer * num_node_per_section, 1)

node_list = []
bc_list = []


# for i in range(0, num_layer):
#     node_list.append(0+201*i)
#     node_list.append(1+201*i)
#     node_list.append(2+201*i)
#     node_list.append(3+201*i)
#     node_list.append(108+201*i)
#     for j in range(0,5):
#         if i == 0:
#             bc_list.append([1,2])
#         else:
#             bc_list.append([-1,-1])

for i in range(0, num_layer):
    node_list.append(0  +201*i)
    node_list.append(1  +201*i)
    node_list.append(2  +201*i)
    node_list.append(3  +201*i)
    node_list.append(7  +201*i)
    node_list.append(13 +201*i)
    node_list.append(20 +201*i)
    node_list.append(27 +201*i)
    node_list.append(35 +201*i)
    node_list.append(62 +201*i)
    node_list.append(96 +201*i)
    node_list.append(108+201*i)
    node_list.append(112+201*i)
    node_list.append(119+201*i)
    node_list.append(128+201*i)
    node_list.append(155+201*i)
    node_list.append(189+201*i)
    for j in range(0,17):
        if i == 0:
            bc_list.append([1,2])
        else:
            bc_list.append([-1,-1])

# print(node_list)
# print('Start Data check!')
# fp = open(fname_check, 'w')
# for i in range(0, num_geo):
#     for j in range(0, num_bc):
#         for k in range(0, num_tstep):
#             fname = workdir + '{geo:0>4d}/{bc:0>4d}/controlmesh_allparticle_{t}.vtk'.format(geo = i + 1, bc = j + 1, t = k)
#             if not path.exists(fname):
#                 print(fname + ' Not exists', file = fp)
#                 break
#         print('{geo:0>4d}/{bc:0>4d} check done!'.format(geo = i+1, bc = j+1))
#     print('{geo:0>4d} check done!'.format(geo = i+1))

# fp.close()

# print('Data check done!')

#%%
# f = h5py.File(workdir_h5+'PipeData.hdf5','w')
with h5py.File(fname_h5,'w') as f:
    f.create_dataset("feature", feature_shape, np.float32)
    f.create_dataset("target", target_shape, np.float32)
    for i in range(geo_start, geo_start + num_geo):
        # f = h5py.File(workdir_h5+'Geo{:0>4d}.hdf5'.format(i),'w')
        # grp = f.create_group('/Geo{:0>4d}'.format(i + 1))
        # grp.create_dataset("feature", feature_shape, np.float32)
        # grp.create_dataset("target", target_shape, np.float32)

        for j in range(0, num_bc):
            vec_bc1 = np.ones((num_node_per_section, 1), dtype = np.float32)
            vec_bc2 = np.zeros(((num_layer - 1) * num_node_per_section, 1), dtype = np.float32)
            vec_bc1 = vec_bc1 * (j + 1) * 0.05
            vec_bc = np.concatenate((vec_bc1, vec_bc2), axis = 0)
            # print(vec_bc)
            for k in range(0, num_tstep):
                fname = workdir + '{geo:0>4d}/{bc:0>4d}/controlmesh_allparticle_{t}.vtk'.format(geo = i + 1, bc = j + 1, t = k + 1)
                if j == 0:
                    pts, vals = utils.ReadAndExtractVTK(fname, j, node_list)
                else:
                    _, vals = utils.ReadAndExtractVTK(fname, j, node_list)
                # print(pts, vals)
                
                vec_t = np.ones((num_layer * num_node_per_section, 1), dtype = np.float32)
                vec_t = vec_t * (k + 1) * 0.1
                # print(vec_t)

                mat_input = np.array(pts)
                mat_input = np.concatenate((pts, vec_t, vec_bc), axis = 1)
                # print(mat_input)

                mat_output = np.array(vals)

                # print(mat_output.shape)

                # grp["feature"][k+j*num_tstep,...] = mat_input
                # grp["target"][k+j*num_tstep,...] = mat_output

                f["feature"][k+j*num_tstep,...] = mat_input
                f["target"][k+j*num_tstep,...] = mat_output

        print("Group {:d} Done!".format(i + 1))

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
