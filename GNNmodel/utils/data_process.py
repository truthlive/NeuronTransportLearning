#%%
import numpy as np
import h5py

# * Read Coarse to Dense mesh mapping for extraction
def ReadC2DMapping(filename):
    C2Dmapping = np.loadtxt(filename, dtype=int, delimiter=",").reshape((-1,))
    # print(C2Dmapping)
    return C2Dmapping


# * Read Coarse to simulator mesh mapping for decomposition and prediction
def ReadC2SMapping(filename):
    C2Smapping = np.loadtxt(filename, dtype=int)
    # print(C2Smapping)
    return C2Smapping


# * Read Dense to simulator mesh mapping for extraction
def ReadS2DMapping(filename):
    S2Dmapping = np.loadtxt(filename, dtype=int, delimiter=",").reshape((-1,))
    # print(S2Dmapping)
    return S2Dmapping


# * Read the edge of the simulator
def ReadSimulatorEdge(filename):
    edge = np.transpose(np.loadtxt(filename, dtype=int))
    return edge


def ReadSimulationParameter(filename):
    fp = open(filename, "r")
    sim_para = {}
    for count, line in enumerate(fp):
        tmp_old = line.split(" ")
        tmp = [elem for elem in tmp_old if elem.strip()]

        sim_para[tmp[0]] = float(tmp[1])

    fp.close()
    return sim_para


# * Read the graph representation of the tree
def ReadGraphRep(filename):

    fp = open(filename, "r")
    idpar = []
    num_sim = 0
    for count, line in enumerate(fp):
        if count > 5:
            line_tmp = np.fromstring(line, dtype="float", sep=" ")
            # print(line_tmp)
            idpar_tmp = (
                line_tmp[[0, 1, -1]].astype("int").tolist()
            )  # ! idx_sim, type, parent_sim
            idpar.append(idpar_tmp)
            num_sim += 1

    fp.close()
    return idpar, num_sim


def GetS2Dmapping(C2Dmapping, C2Smapping):
    npt_extract = int(C2Smapping.shape[0])
    S2Dmapping = np.empty([npt_extract,], dtype=int)
    for i in range(npt_extract):
        S2Dmapping[C2Smapping[i, 1]] = C2Dmapping[C2Smapping[i, 0]]
    return S2Dmapping


def ExtractLineList(S2Dmapping, npt, nele):
    # ! work for S2Dmapping and C2Dmapping
    n_line_before_pt = 5
    n_line_before_ptdata = n_line_before_pt + npt + 2 * (nele + 2) + 3
    npt_extract = int(S2Dmapping.shape[0])

    line_xyz = np.empty([npt_extract,], dtype=int)
    line_data = np.empty([npt_extract,], dtype=int)

    for i in range(npt_extract):
        line_xyz[i] = S2Dmapping[i] + n_line_before_pt
        line_data[i] = S2Dmapping[i] + n_line_before_ptdata

    return line_xyz, line_data


def ReadAndExtractVTK(filename, node_extract_list, flag_xyz, flag_vals=0):
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
    # with open(filename) as fp:
    fp = open(filename, "r")
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

            if (
                count >= val_start
                and count < val_end
                and flag_vals == 0
                and val_it < num_extract
            ):
                if count - val_start == node_extract_list[val_it]:
                    tmp_vals = np.fromstring(line, dtype=float, sep=" ")
                    vals.append(tmp_vals.tolist())
                    val_it += 1

        if count >= val_end and val_end > 0:
            fp.close()
            return pts, vals

    return pts, vals


""" def ReadAndExtractVTK_List(filename, line_xyz, line_vals, flag_xyz, flag_vals=0):

    num_extract = line_xyz.shape[0]
    # print(num_extract)

    pts = []
    vals = []
    xyz_it = 0
    vals_it = 0

    # with open(filename ,"r") as fp:
    fp = open(filename, "r")
    for count, line in enumerate(fp):
        if len(pts) >= num_extract and len(vals) >= num_extract:
            fp.close()
            break
        if count in line_xyz and flag_xyz == 0:
            tmp_xyz = np.fromstring(line, dtype=float, sep=" ")
            pts.append(tmp_xyz.tolist())
            xyz_it += 1
        if count in line_vals and flag_vals == 0:
            tmp_vals = np.fromstring(line, dtype=float, sep=" ")
            vals.append(tmp_vals.tolist())
            vals_it += 1

    return pts, vals """


def ReadAndExtractVTK_List(filename, line_xyz, line_vals, flag_xyz, flag_vals=0):
    num_extract = line_xyz.shape[0]
    # print(num_extract)

    pts = np.empty([num_extract, 3], dtype=float)
    vals = np.empty([num_extract, 1], dtype=float)
    xyz_it = 0
    vals_it = 0

    sort_xyz = np.argsort(line_xyz)
    sort_vals = np.argsort(line_vals)

    # with open(filename ,"r") as fp:
    fp = open(filename, "r")
    for count, line in enumerate(fp):
        if xyz_it >= num_extract and vals_it >= num_extract:
            fp.close()
            break

        if xyz_it < num_extract and flag_xyz == 0:
            if count == line_xyz[sort_xyz[xyz_it]]:
                tmp_xyz = np.fromstring(line, dtype=float, sep=" ")
                pts[sort_xyz[xyz_it], :] = tmp_xyz
                xyz_it += 1

        if vals_it < num_extract and flag_vals == 0:
            if count == line_vals[sort_vals[vals_it]]:
                tmp_vals = np.fromstring(line, dtype=float, sep=" ")
                vals[sort_vals[vals_it], :] = tmp_vals
                vals_it += 1

    return pts, vals


def EncodePipeGeo(pts, theta_template, num_layer, num_node_layer):

    coor_xyz = np.asarray(pts)
    theta_all = np.asarray(theta_template)

    # print(coor_xyz)
    # print(type(coor_xyz))

    coor_cylinder = []
    s_current = 0

    for i in range(num_layer):
        start = int(i * num_node_layer)
        end = int((i + 1) * num_node_layer)
        # print(start, end)
        # print(coor_xyz[start:end, :] - coor_xyz[start, :])

        vec_r = coor_xyz[start:end, :] - coor_xyz[start, :]
        r = np.linalg.norm(vec_r, axis=1)

        if i == 0:
            s = np.zeros([num_node_layer,], dtype=float)
        else:
            s_current += np.linalg.norm(
                coor_xyz[start, :] - coor_xyz[start - num_node_layer, :]
            )
            s = np.ones([num_node_layer,], dtype=float) * s_current

        theta = theta_all[start:end]
        # print(r.shape)
        # print(theta.shape)
        # print(s.shape)
        coor_cylinder += np.stack((r, theta, s), axis=-1).tolist()
        # print(coor_cylinder)
        # r = coor_xyz[start:end, :] - coor_cylinder[start, :]
        # r = np.linalg.norm(coor_xyz[start:end, :] - coor_cylinder[start, :])
        # print(r)
        # r =
        # for j in range(num_node_layer):
        #     r = sqrt
    return coor_cylinder


# # * fname_C2S: Variable
# model_name = "1bifurcation"
# fname_meshinfo = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/controlmesh_info.txt"
# fname_C2D = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/C2Dmapping_extract.txt"
# fname_C2S = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/Simulator/simulator_1_C2Smapping.txt"
# fname_S2D = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/Simulator/simulator_1_S2Dmapping.txt"
# fname_edge = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/Simulator/simulator_1_edge.txt"
# # fname_edge = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/pipe_graph_topo_17_local.txt"
# C2Dmapping = ReadC2DMapping(fname_C2D) # mapping from coarse mesh to dense mesh
# C2Smapping = ReadC2SMapping(fname_C2S) # mapping between Coarse mesh and each simulator
# # S2Dmapping = GetS2Dmapping(C2Dmapping, C2Smapping) # Dense mesh to each simulator

""" # ! Extract and generate dataset
model_name = "1bifurcation"
fname_meshinfo = (
    "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"
    + model_name
    + "/controlmesh_info.txt"
)
fname_S2D = (
    "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"
    + model_name
    + "/Simulator/simulator_1_S2Dmapping.txt"
)
fname_edge = (
    "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"
    + model_name
    + "/Simulator/simulator_1_edge.txt"
)
fname_data = (
    "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"
    + model_name
    + "/controlmesh.vtk"
)

num_layer = 8


S2Dmapping = ReadS2DMapping(fname_S2D)  # mapping from simulator mesh to dense mesh
num_pt, num_ele = np.loadtxt(fname_meshinfo, dtype=int)
print(num_pt, num_ele)
line_xyz, line_vals = ExtractLineList(S2Dmapping, num_pt, num_ele)
edge = ReadSimulatorEdge(fname_edge)
pts, vals = ReadAndExtractVTK_List(fname_data, line_xyz, line_vals, 0)
print(len(pts))

num_node_layer = int(len(pts) / num_layer)

theta_template = []
for i in range(num_layer):
    theta_template.append(0.0 / 180.0 * np.pi)
    theta_template.append(180.0 / 180.0 * np.pi)
    theta_template.append(90.0 / 180.0 * np.pi)
    theta_template.append(0.0 / 180.0 * np.pi)
    theta_template.append(180.0 / 180.0 * np.pi)
    theta_template.append(90.0 / 180.0 * np.pi)
    theta_template.append(135.0 / 180.0 * np.pi)
    theta_template.append(0.0 / 180.0 * np.pi)
    theta_template.append(45.0 / 180.0 * np.pi)
    theta_template.append(135.0 / 180.0 * np.pi)
    theta_template.append(45.0 / 180.0 * np.pi)
    theta_template.append(270.0 / 180.0 * np.pi)
    theta_template.append(270.0 / 180.0 * np.pi)
    theta_template.append(225.0 / 180.0 * np.pi)
    theta_template.append(315.0 / 180.0 * np.pi)
    theta_template.append(225.0 / 180.0 * np.pi)
    theta_template.append(315.0 / 180.0 * np.pi)

pts_cylinder = EncodePipeGeo(pts, theta_template, num_layer, num_node_layer)
print(pts_cylinder)
print(type(pts_cylinder)) """

#%%

# fname = "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/"+ model_name +"/0001/controlmesh_allparticle_1.vtk"

# # %%
# import numpy as np
# num_layer = 8
# fname_S2Dmapping= "/pylon5/eg560mp/angranl/NeuronMachineLearning/MLdata/test4_PipeNew/S2Dmapping.txt"
# node_list = []
# for i in range(0, num_layer):
#     node_list.append(0  +201*i)
#     node_list.append(1  +201*i)
#     node_list.append(2  +201*i)
#     node_list.append(3  +201*i)
#     node_list.append(7  +201*i)
#     node_list.append(13 +201*i)
#     node_list.append(20 +201*i)
#     node_list.append(27 +201*i)
#     node_list.append(35 +201*i)
#     node_list.append(62 +201*i)
#     node_list.append(96 +201*i)
#     node_list.append(108+201*i)
#     node_list.append(112+201*i)
#     node_list.append(119+201*i)
#     node_list.append(128+201*i)
#     node_list.append(155+201*i)
#     node_list.append(189+201*i)
#     # for j in range(0,17):
#     #     if i == 0:
#     #         bc_list.append([1,2])
#     #     else:
#     #         bc_list.append([-1,-1])

# np.savetxt(fname_S2Dmapping, node_list, fmt = "%d", newline= ",")
