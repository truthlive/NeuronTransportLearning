import numpy as np
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
from torch_scatter import scatter_add

def make_mlp(input_dim, hidden_dim, output_dim, layer_num, activation='ReLU',
             final_activation=False, batchnorm=None):
    # assert layer_num >= 2
    activation_layer_func = getattr(nn, activation)
    mlp_layers = [nn.Linear(input_dim, hidden_dim),]
    # if batchnorm == 'LayerNorm':
    #     mlp_layers.append(nn.LayerNorm(hidden_dim))
    # elif batchnorm == 'BatchNorm':
    #     mlp_layers.append(nn.BatchNorm1d(hidden_dim))
    mlp_layers.append(activation_layer_func())
    if layer_num > 1:
        for li in range(1, layer_num - 1):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            # if batchnorm == 'LayerNorm':
            #     mlp_layers.append(nn.LayerNorm(hidden_dim))
            # elif batchnorm == 'BatchNorm':
            #     mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(activation_layer_func())
    mlp_layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation:
        if batchnorm == 'LayerNorm':
            mlp_layers.append(nn.LayerNorm(hidden_dim))
        elif batchnorm == 'BatchNorm':
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
        mlp_layers.append(activation_layer_func())
    return nn.Sequential(*mlp_layers)

def update_graph(data, **kwargs):
    '''
    Create a new Data filled with kwargs, use features not listed in kwargs from data
    :param data: a torch_geometric.data.Data struct. They are not cloned!
    :param kwargs: features to replace. They are not cloned.
    :return: a new Data structure filled with kwargs and other features from data
    '''
    new_data_dict = {}
    for k, v in data:
        if k not in kwargs.keys():
            new_data_dict[k] = v
    for k, v in kwargs.items():
        new_data_dict[k] = v
    new_data = Data.from_dict(new_data_dict)
    return new_data

def decompose_graph(graph):
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)
