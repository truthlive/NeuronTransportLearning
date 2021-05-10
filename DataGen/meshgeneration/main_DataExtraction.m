%% This file is used to create the geometry (hex mesh) for the ML dataset
% Pipeline: Tree skeletion -> Tree hex mesh -> Extract pipe/bif mesh

%% (Must run) Skeleton extraction for training and prediction
% Create the skeleton information of each pipe and bifurcation
% The skeleton information is then used for extract the mesh of pipe or bif
clc;
clear;
addpath(genpath(pwd));
start_trees;

num_pipe = 1;
num_bifur = 1;

% Extract the graph with fixed pipe structure 
% io_path = '..//..//MLdata//3bifurcation//';
% io_path = '..//..//MLdata//1bifurcation//';
% io_path = '..//..//MLdata//bifurcation_old//';
% io_path =  '..//..//MLdata//1bifurcation_test//';
% io_path = '..//..//MLdata//NMO_66748//';
% ExtractGraph_FixPipe(io_path);

% Extract the graph from old complex tree (The bspline smooth function works differently)
% io_path = '..//..//MLdata//3bifurcation_old//';
io_path = '..//..//MLdata//NMO_66748//';
% io_path = '..//..//MLdata//NMO_66731//';

ExtractGraph_oldComplexTree(io_path);

% Extract the graph (the num of cross sections in a pipe is relaxed)
% io_path = '..//..//MLdata//3bifurcation_test//';
% ExtractGraph(io_path);
trees{3} = load_tree([io_path,  'GraphRep.swc']);
figure(3); xplore_tree(trees{3})

%% (Training Data) Generate tree hex mesh and extract pipe and bifurcation mesh for simulation
% io_path = '..//..//MLdata//NMO_66748//';
% n_pipe_start = 0;
% n_bif_start = 0;
io_path = '..//..//MLdata//NMO_66731//';
n_pipe_start = 69;
n_bif_start = 34;
GenHexMesh(io_path, n_pipe_start, n_bif_start);
%% (Training Data) Generate the mapping between simulation mesh and ML training graph
% io_path = '..//..//MLdata//NMO_66748//';
% n_pipe_start = 0;
% n_bif_start = 0;
io_path = '..//..//MLdata//NMO_66731//';
n_pipe_start = 69;
n_bif_start = 34;
GenHexMeshToSimulator(io_path, n_pipe_start, n_bif_start);

%% (Prediction Data) Generate Tree hexmesh and the mapping between tree mesh and ML prediction graph
% io_path =  '..//..//MLdata//1bifurcation_test//';
% io_path =  '..//..//MLdata//3bifurcation_test//';
% io_path = '..//..//MLdata//3bifurcation_old//';
io_path = '..//..//MLdata//NMO_66748//';
% io_path = '..//..//MLdata//NMO_66731//';
GenHexGraph(io_path);
