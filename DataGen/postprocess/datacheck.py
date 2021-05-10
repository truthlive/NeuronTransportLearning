
import os
# import utils
import h5py
import numpy as np
import os.path
from os import path 

import h5py
workdir = '/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/'
workdir_h5 = workdir + 'HDF5/'
fname_h5 = workdir_h5 + 'PipeDataAll_17_{}.h5'.format(2)
with h5py.File(fname_h5, 'r') as f_debug:
    # print(f_debug["feature"][1, :, :])
    a = f_debug["feature"][1, :, :]
    print(a)
    # for gname in f_debug.keys():
    # #     print(f_debug[gname].keys())
    #     print(f_debug[gname]['feature'][3,:,:])

num_geo = 100
num_bc = 200
num_tstep = 100
num_layer = 8
num_node_per_section = 17
workdir = '/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/'
workdir_h5 = workdir + 'HDF5/'
fname_h5 = workdir_h5 + 'PipeDataAll_17.h5'
fname_check = workdir_h5 + 'data_check_new.txt'

feature_shape = (num_bc*num_tstep, num_layer * num_node_per_section, 5)
target_shape = (num_bc*num_tstep, num_layer * num_node_per_section, 1)

node_list = []
bc_list = []


# print(node_list)
print('Start Data check!')
# fp = open(fname_check, 'w')
with open(fname_check, 'a+') as fp:
    for i in range(87, num_geo):
        
        j_start = 0
        if i == 87:
            j_start = 61
        for j in range(j_start, num_bc):
            myPath = workdir + '{geo:0>4d}/{bc:0>4d}/'.format(geo = i + 1, bc = j + 1)
            tmp_len = len([f for f in os.listdir(myPath) 
        if f.startswith('controlmesh_allparticle_') and os.path.isfile(os.path.join(myPath, f))])
            print(tmp_len)
            print(myPath + ' match file number: {:d}'.format(tmp_len), file = fp)



            # for k in range(0, num_tstep):
            #     fname = workdir + '{geo:0>4d}/{bc:0>4d}/controlmesh_allparticle_{t}.vtk'.format(geo = i + 1, bc = j + 1, t = k)
            #     if not path.exists(fname):
            #         print(fname + ' Not exists', file = fp)
            #         break
            print('{geo:0>4d}/{bc:0>4d} check done!'.format(geo = i+1, bc = j+1))
        print('{geo:0>4d} check done!'.format(geo = i+1))

# fp.close()

