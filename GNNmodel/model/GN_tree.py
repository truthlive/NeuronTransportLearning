# import torch

# import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn import (
    SplineConv,
    SAGEConv,
    GCNConv,
    MetaLayer,
)

from model.GN_simulator import SimulatorModel

# from blocks import EdgeBlock, NodeBlock, GlobalBlock


class AssemblyModel(torch.nn.Module):
    def __init__(self, net=None):
        super(AssemblyModel, self).__init__()
        self.net = net

    def _forward_one_net(self, net, src, dest):

        x_src, x_dest = src, dest
        disp_r = x_dest[:, :3] - x_src[:, :3]
        f_r = x_dest[:, -1] - x_src[:, -1]
        f_r = f_r.reshape((-1, 1))
        f_r3 = torch.cat((f_r, f_r, f_r), dim=-1)
        grad_r = torch.div(f_r3, disp_r)
        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))

        if net is None:

            # out = torch.cat((grad_r, norm_r), dim=-1)
            out = torch.cat((disp_r, norm_r, f_r), dim=-1)
            # out = x_dest - x_src
        else:
            # net_input = torch.cat((grad_r, norm_r), dim=-1)
            net_input = torch.cat((disp_r, norm_r, f_r), dim=-1)

            out = net(net_input)

        return out

    def forward(self, src, dest, edge_attr, u, batch):

        return self._forward_one_net(self.net, src, dest)


class GN_tree(torch.nn.Module):
    def __init__(self, sim_net, num_layers, num_hidden):
        # def __init__(self):
        super(GN_tree, self).__init__()

        self.num_features = dataset.feature.shape[-1]
        self.num_targets = dataset.target.shape[-1]

        # ! New GN
        self.mlp1 = nn.Sequential(
            nn.Linear(5, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 5),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_features + 2, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, self.num_features),
        )

        # self.eb = EdgeModel(net=self.mlp1)
        self.eb = AssemblyModel()
        # self.eb = EdgeGradientLayer()
        self.nb = sim_net
        self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(self.gn)

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

        self.pipe_conv1 = SAGEConv(self.num_features, num_hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(num_hidden, num_hidden))
        self.pipe_conv2 = SAGEConv(num_hidden, self.num_targets)

    def forward(self, data):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index

        for processor in self.processors:
            x_res = processor(x, edge_index)[0]
            x = x + F.relu(x_res)
        out = self.decoder(x)

        return out
