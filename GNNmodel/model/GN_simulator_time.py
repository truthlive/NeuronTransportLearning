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

# from blocks import EdgeBlock, NodeBlock, GlobalBlock


class EdgeModel(torch.nn.Module):
    def __init__(self, net=None):
        super(EdgeModel, self).__init__()
        self.net = net

    def _forward_one_net(self, net, src, dest, edge_attr):
        disp_r = dest[:, :3] - src[:, :3]
        f_src = src[:, -1].reshape((-1, 1))
        f_dest = dest[:, -1].reshape((-1, 1))
        # f_r = dest[:, -1] - src[:, -1]
        # f_r = f_r.reshape((-1, 1))
        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))

        # print()
        # print(edge_attr.shape)

        net_in = torch.cat([disp_r, norm_r, edge_attr, f_src, f_dest], dim=-1)
        # print(net_in.shape)
        # net_in = torch.cat([net_in, edge_attr], dim=-1)
        # print(net_in.shape)
        out = net(net_in)
        out = edge_attr + out

        return out

    def forward(self, src, dest, edge_attr, u, batch):
        return self._forward_one_net(self.net, src, dest, edge_attr)


class NodeModel(MessagePassing):
    def __init__(self, net=None, aggr="mean", flow="source_to_target"):
        super(NodeModel, self).__init__(aggr=aggr)
        self.net = net

    def _message_one_net(self, net, x_i, x_j, edge_attr):
        return edge_attr

    def message(self, x_i, x_j, edge_attr):
        return self._message_one_net(self.net, x_i, x_j, edge_attr)

    def update(self, aggr_out, x):
        # net_input = torch.cat((x[:, -1].unsqueeze(-1), aggr_out), dim=-1)
        net_input = torch.cat((x[:, -2:], aggr_out), dim=-1)
        net_out = self.net(net_input)
        out = x
        out[:, -1] = out[:, -1] + net_out.squeeze()

        return out

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr
        )


# class EdgeModel(torch.nn.Module):
#     def __init__(self, net=None):
#         super(EdgeModel, self).__init__()
#         self.net = net

#     def _forward_one_net(self, net, src, dest, edge_attr):

#         out = net(edge_attr)
#         out = edge_attr + out

#         return out

#     def forward(self, src, dest, edge_attr, u, batch):

#         return self._forward_one_net(self.net, src, dest, edge_attr)

# class NodeModel(MessagePassing):
#     def __init__(self, net=None, aggr="mean", flow="source_to_target"):
#         super(NodeModel, self).__init__(aggr=aggr)
#         self.net = net

#     def _message_one_net(self, net, x_i, x_j, edge_attr):
#         if net is None:
#             return x_i - x_j
#         else:
#             return edge_attr

#     def message(self, x_i, x_j, edge_attr):
#         return self._message_one_net(self.net, x_i, x_j, edge_attr)

#     def update(self, aggr_out, x):
#         net_input = torch.cat((aggr_out, x[:, 4:]), dim=-1)
#         out = self.net(net_input)
#         return out

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         return self.propagate(
#             edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr
#         )


# ! GN simulator
class Simulator(torch.nn.Module):
    def __init__(self, dataset, num_layers, num_hidden):
        super(Simulator, self).__init__()

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
        self.eb = EdgeModel()
        # self.eb = EdgeGradientLayer()
        self.nb = NodeModel(net=self.mlp2)
        self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(self.gn)

        # Not Shared parameter
        # for i in range(num_layers - 1):
        #    self.processors.append(MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(net=self.mlp2)))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

    def forward(self, data, mode):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index

        # print(edge_attr.shape)

        if mode == 1:
            # ! New GEN + Decoder

            for processor in self.processors:
                x_res = processor(x, edge_index)[0]
                x = x + F.relu(x_res)
            out = self.decoder(x)

        return out
        # return x


# ! GN Simulator change init
class SimulatorModel(torch.nn.Module):
    def __init__(self, num_features, num_targets, num_layers, num_hidden):
        # def __init__(self):
        super(SimulatorModel, self).__init__()

        self.num_features = num_features
        self.num_targets = num_targets

        # ! New GN
        self.eb = EdgeModel(
            net=nn.Sequential(
                nn.Linear(3 + 6, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 3),
            )
        )
        self.nb = NodeModel(
            net=nn.Sequential(
                nn.Linear(5, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 1),
            )
        )

        # self.eb = EdgeModel(
        #     net=nn.Sequential(
        #         nn.Linear(3 + 6, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, 3),
        #     )
        # )
        # self.nb = NodeModel(
        #     net=nn.Sequential(
        #         nn.Linear(5, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, 1),
        #     )
        # )
        self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(MetaLayer(edge_model=self.eb, node_model=self.nb))

        # Not Shared parameter
        # for i in range(num_layers - 1):
        #    self.processors.append(MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(net=self.mlp2)))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features - 1, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

    def _compute_edge_attr(self, x, edge_index):
        x_src = x[edge_index[0, :], :]
        x_dest = x[edge_index[1, :], :]

        disp_r = x_dest[:, :3] - x_src[:, :3]
        f_r = x_dest[:, -1] - x_src[:, -1]
        f_r = f_r.reshape((-1, 1))
        f_r3 = torch.cat((f_r, f_r, f_r), dim=-1)

        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))
        grad_r = torch.mul(f_r3, disp_r)
        # grad_r = torch.div(grad_r, norm_r)

        # print(disp_r.shape)
        # print(f_r.shape)
        # print(norm_r.shape)

        # edge_attr = torch.cat((disp_r, norm_r, f_r), dim=-1)

        # disp_r = torch.div(disp_r, norm_r)
        # grad_r = torch.div(f_r, norm_r)
        # edge_attr = torch.cat((disp_r, norm_r, grad_r), dim=-1)
        # edge_attr = torch.cat((norm_r, f_r, grad_r), dim=-1)
        edge_attr = grad_r

        # print(edge_attr.shape)
        # print(edge)

        return edge_attr

    def forward(self, X_curr, edge, y_prev, mode):
        # x, edge_index, edge_attr = X_k.x, X_k.edge_index, X_k.edge_attr
        x, edge_index = X_curr, edge

        x_in = torch.cat((x[:, [0, 1, 2, 4]], y_prev.unsqueeze(-1)), dim=-1)
        # x_in = torch.cat((x[:, [0, 1, 2]], y_prev.unsqueeze(-1)), dim=-1)
        # x_in = torch.cat((x, y_prev.unsqueeze(-1)), dim=-1)
        edge_attr = self._compute_edge_attr(x_in, edge_index)
        for processor in self.processors:
            x_in, edge_attr, _ = processor(x_in, edge_index, edge_attr)
            # x_res, edge_attr, _ = processor(x_in, edge_index, edge_attr)
            # edge_attr = edge_attr + F.relu(edge_attr_res)s
            # x_in = x_in + F.relu(x_res)
        x_out = self.decoder(x_in)

        out = y_prev + x_out.squeeze()

        return out


# class SimulatorModel(torch.nn.Module):
#     def __init__(self, num_features, num_targets, num_layers, num_hidden):
#         # def __init__(self):
#         super(SimulatorModel, self).__init__()

#         self.num_features = num_features
#         self.num_targets = num_targets

#         # ! New GN
#         self.mlp_node = nn.Sequential(
#             nn.Linear(self.num_features + 1, num_hidden),
#             nn.ReLU(),
#             nn.Linear(num_hidden, num_hidden),
#             nn.ReLU(),
#             nn.Linear(num_hidden, self.num_features),
#         )

#         self.eb = EdgeModel(
#             net=nn.Sequential(
#                 nn.Linear(5, num_hidden),
#                 nn.ReLU(),
#                 nn.Linear(num_hidden, num_hidden),
#                 nn.ReLU(),
#                 nn.Linear(num_hidden, 5),
#             )
#         )
#         self.nb = NodeModel(
#             net=nn.Sequential(
#                 nn.Linear(self.num_features + 1, num_hidden),
#                 nn.ReLU(),
#                 nn.Linear(num_hidden, num_hidden),
#                 nn.ReLU(),
#                 nn.Linear(num_hidden, self.num_features),
#             )
#         )
#         self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

#         self.processors = torch.nn.ModuleList()

#         # Shared parameter
#         for i in range(num_layers - 1):
#             self.processors.append(MetaLayer(edge_model=self.eb, node_model=self.nb))

#         # Not Shared parameter
#         # for i in range(num_layers - 1):
#         #    self.processors.append(MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(net=self.mlp2)))

#         self.decoder = nn.Sequential(
#             nn.Linear(self.num_features, num_hidden),
#             nn.ReLU(True),
#             nn.Linear(num_hidden, num_hidden),
#             nn.ReLU(True),
#             nn.Linear(num_hidden, num_hidden),
#             nn.ReLU(True),
#             nn.Linear(num_hidden, self.num_targets),
#         )

#     def _compute_edge_attr(self, x, edge_index):
#         x_src = x[edge_index[0, :], :]
#         x_dest = x[edge_index[1, :], :]

#         disp_r = x_dest[:, :3] - x_src[:, :3]
#         f_r = x_dest[:, -1] - x_src[:, -1]
#         f_r = f_r.reshape((-1, 1))
#         norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))

#         print(disp_r.shape)
#         print(f_r.shape)
#         print(norm_r.shape)

#         # edge_attr = torch.cat((disp_r, norm_r, f_r), dim=-1)

#         disp_r = torch.div(disp_r, norm_r)
#         grad_r = torch.div(f_r, norm_r)
#         edge_attr = torch.cat((disp_r, norm_r, grad_r), dim=-1)

#         print(edge_attr.shape)
#         # print(edge)

#         return edge_attr

#     def forward(self, X_curr, edge, y_prev, mode):
#         # x, edge_index, edge_attr = X_k.x, X_k.edge_index, X_k.edge_attr
#         x, edge_index = X_curr, edge

#         x_in = torch.cat((x, y_prev.unsqueeze(-1)), dim=-1)
#         edge_attr = self._compute_edge_attr(x_in, edge_index)
#         for processor in self.processors:
#             x_res = processor(x_in, edge_index, edge_attr)[0]
#             x_in = x_in + F.relu(x_res)
#         x_out = self.decoder(x_in)
#         # print(x_out.shape)
#         out = y_prev + x_out.squeeze()
#         # out = y_prev + self.decoder(x_in)

#         # # print(edge_attr.shape)

#         # if mode == 1:
#         #     for processor in self.processors:
#         #         x_res = processor(x, edge_index)[0]
#         #         x = x + F.relu(x_res)
#         #     out = self.decoder(x)
#         return out
