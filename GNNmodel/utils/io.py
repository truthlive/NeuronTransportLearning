import numpy as np


def WriteVTK(filename, pts, val_node, edge_index):

    nPoint = 0
    nElement = 0

    # fn_output_edge = "./data/Pipe/pipe_graph_topo_17_local.txt"
    # edge = np.loadtxt(fn_output_edge, delimiter="\t", dtype=int)
    edge = edge_index
    # edge = add_ghost_edge(edge, 136, 17)

    nPoint += len(pts)
    nElement += len(edge)

    outF = open(filename, "w")
    outF.write("# vtk DataFile Version 4.2\n")
    outF.write("vtk output\n")
    outF.write("ASCII\n")
    outF.write("DATASET UNSTRUCTURED_GRID\n")
    print("POINTS {} double".format(nPoint), file=outF)

    for x in pts.to("cpu").tolist():
        print(*x, file=outF, sep=" ")

    # for i in range(0, len(pts)):
    #     print(" ".join(str(x) for x in pts[i]), file=outF)

    print("CELLS {} {}".format(nElement, nElement * 3), file=outF)
    for i in range(0, len(edge)):
        print("2", " ".join(str(int(x)) for x in edge[i]), file=outF)

    print("CELL_TYPES {}".format(nElement), file=outF)
    for i in range(0, nElement):
        outF.write("3\n")

    print("POINT_DATA {}".format(nPoint), file=outF)
    print("SCALARS AllParticles float 1", file=outF)
    print("LOOKUP_TABLE default", file=outF)

    for x in val_node.to("cpu").tolist():
        # print(*x, file=outF, sep=" ")
        print(x, file=outF, sep=" ")

    outF.close()


def WriteVTK_bcini(filename, pts, val_node):

    nPoint = 0
    nElement = 0

    fn_output_edge = "./data/Pipe/pipe_graph_topo_17_local.txt"
    edge = np.loadtxt(fn_output_edge, delimiter="\t", dtype=int)
    # edge = add_ghost_edge(edge, 136, 17)

    nPoint += len(pts)
    nElement += len(edge)

    outF = open(filename, "w")
    outF.write("# vtk DataFile Version 4.2\n")
    outF.write("vtk output\n")
    outF.write("ASCII\n")
    outF.write("DATASET UNSTRUCTURED_GRID\n")
    print("POINTS {} double".format(nPoint), file=outF)

    for x in pts.to("cpu").tolist():
        print(*x, file=outF, sep=" ")
    # for i in range(0, len(pts)):
    #     print(" ".join(str(x) for x in pts[i]), file=outF)

    print("CELLS {} {}".format(nElement, nElement * 3), file=outF)
    for i in range(0, len(edge)):
        print("2", " ".join(str(int(x)) for x in edge[i]), file=outF)

    print("CELL_TYPES {}".format(nElement), file=outF)
    for i in range(0, nElement):
        outF.write("3\n")

    print("POINT_DATA {}".format(nPoint), file=outF)
    print("SCALARS AllParticles float 1", file=outF)
    print("LOOKUP_TABLE default", file=outF)

    for x in val_node.to("cpu").tolist():
        print(x, file=outF, sep=" ")

    outF.close()
