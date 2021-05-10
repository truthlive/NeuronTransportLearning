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
    GMMConv,
)

# ! Original Modules for Graph Network
class EdgeModel(torch.nn.Module):
    def __init__(self, net=None):
        super(EdgeModel, self).__init__()
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
            out = torch.cat((disp_r, norm_r, f_r), dim=-1)
        else:
            net_input = torch.cat((disp_r, norm_r, f_r), dim=-1)

            out = net(net_input)

        return out

    def forward(self, src, dest, edge_attr, u, batch):

        return self._forward_one_net(self.net, src, dest)


class NodeModel(MessagePassing):
    def __init__(self, net=None, aggr="mean", flow="source_to_target"):
        super(NodeModel, self).__init__(aggr=aggr)
        self.net = net

    def _message_one_net(self, net, x_i, x_j, edge_attr):
        if net is None:
            return x_i - x_j
        else:
            return edge_attr

    def message(self, x_i, x_j, edge_attr):
        return self._message_one_net(self.net, x_i, x_j, edge_attr)

    def update(self, aggr_out, x):
        net_input = torch.cat((aggr_out, x[:, 3:]), dim=-1)
        out = self.net(net_input)

        return out

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr
        )


# ! Modified Modules for GN
class EdgeModel(torch.nn.Module):
    def __init__(self, net=None):
        super(EdgeModel, self).__init__()
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
            out = torch.cat((disp_r, norm_r, f_r), dim=-1)
        else:
            net_input = torch.cat((disp_r, norm_r, f_r), dim=-1)

            out = net(net_input)

        return out

    def forward(self, src, dest, edge_attr, u, batch):

        return self._forward_one_net(self.net, src, dest)


class NodeModel(MessagePassing):
    def __init__(self, net=None, aggr="mean", flow="source_to_target"):
        super(NodeModel, self).__init__(aggr=aggr)
        self.net = net

    def _message_one_net(self, net, x_i, x_j, edge_attr):
        if net is None:
            return x_i - x_j
        else:
            return edge_attr

    def message(self, x_i, x_j, edge_attr):
        return self._message_one_net(self.net, x_i, x_j, edge_attr)

    def update(self, aggr_out, x):
        net_input = torch.cat((aggr_out, x[:, 3:]), dim=-1)
        out = self.net(net_input)

        return out

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr
        )


# ! GN simulator
class Simulator(torch.nn.Module):
    def __init__(self, dataset, num_layers, num_hidden):
        # def __init__(self):
        super(Simulator, self).__init__()

        self.num_features = dataset.feature.shape[-1]
        self.num_targets = dataset.target.shape[-1]

        # ! Encoder + GN + Decoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.num_targets * 2, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, num_hidden),  # b, 84, 11, 11
        #     nn.ReLU(True),
        #     nn.Linear(num_hidden, self.num_targets),  # b, 84, 11, 11
        # )

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(num_hidden * 2, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(num_hidden * 2, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        # )
        # self.eb = EdgeGradientLayer(net=self.mlp1)
        # self.nb = NodeLaplacianLayer(net=self.mlp2)
        # self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        # ! Pure GN
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(self.num_features, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, self.num_features),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(self.num_features * 2, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, self.num_targets),
        # )

        # self.eb = EdgeGradientLayer(net=self.mlp1)
        # # self.eb = EdgeGradientLayer()
        # self.nb = NodeLaplacianLayer(net=self.mlp2)
        # self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

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

    def _compute_edge_attr(self, x, edge_index):
        x_src = x[edge_index[0, :], :]
        x_dest = x[edge_index[1, :], :]

        disp_r = x_dest[:, :3] - x_src[:, :3]
        f_r = x_dest[:, -1] - x_src[:, -1]
        f_r = f_r.reshape((-1, 1))
        f_r3 = torch.cat((f_r, f_r, f_r), dim=-1)

        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))
        grad_r = torch.mul(f_r3, disp_r)
        edge_attr = grad_r

        return edge_attr

    def forward(self, data, mode):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index
        # edge_attr = self._compute_edge_attr(x, edge_index)
        # print(edge_attr.shape)

        if mode == 1:
            # ! GEN

            # x = self.gn(x, edge_index)

            # ! Encoder + GEN + Decoder
            # x = self.encoder(x)
            # x = F.relu(x)
            # x = F.normalize(x, p=2, dim=-1)
            # x = x + self.gn(x, edge_index)[0]
            # x = F.relu(x)
            # x = F.normalize(x, p=2, dim=-1)
            # # for conv in self.pipe_convs:
            # #     x = F.relu(conv(x, edge_index))
            # x = self.decoder(x)
            # # x = F.relu(x)

            # ! New GEN + Decoder
            for processor in self.processors:
                x_res = processor(x, edge_index)[0]
                x = x + F.relu(x_res)
            out = self.decoder(x)

            # for i in range(5):
            #     x_mid = x
            #     for processor in self.processors:
            #         x_res = processor(x_mid, edge_index)[0]
            #         x_mid = x_mid + F.relu(x_res)
            #     x_mid = self.decoder(x_mid)
            #     x[:, -1] = x_mid[:, 0]
            # out = x_mid
            # ! GraphSAGE

            # x = self.pipe_conv1(x, edge_index)
            # x = x + F.relu(x)
            # for conv in self.convs:
            #     x = x + F.relu(conv(x, edge_index))
            # x = self.pipe_conv2(x, edge_index)
            # x = x + F.relu(x)
        return out
        # return x


# ! GN Simulator change init
class SimulatorModel(torch.nn.Module):
    def __init__(self, num_features, num_targets, num_layers, num_hidden):
        # def __init__(self):
        super(SimulatorModel, self).__init__()

        self.num_features = num_features
        self.num_targets = num_targets

        # # ! Old GN
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(5, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, 5),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(self.num_features + 2, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, self.num_features),
        # )

        # self.eb = EdgeModel()
        # self.nb = NodeModel(net=self.mlp2)
        # self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        # ! New GN
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(5, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, 5),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(self.num_features + 2, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, self.num_features),
        # )

        self.eb = EdgeModel(
            nn.Sequential(
                nn.Linear(5, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 5),
            )
        )
        self.nb = NodeModel(
            nn.Sequential(
                nn.Linear(self.num_features + 2, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, self.num_features),
            )
        )
        # self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # # Shared parameter
        # for i in range(num_layers - 1):
        #     self.processors.append(self.gn)

        # Not Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(MetaLayer(edge_model=self.eb, node_model=self.nb))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

        # self.pipe_conv1 = SAGEConv(self.num_features, num_hidden)
        # self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     self.convs.append(SAGEConv(num_hidden, num_hidden))
        # self.pipe_conv2 = SAGEConv(num_hidden, self.num_targets)

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

        return edge_attr

    def forward(self, data, mode):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index

        # print(edge_attr.shape)
        edge_attr = self._compute_edge_attr(x, edge_index)

        if mode == 1:

            # ! New GEN + Decoder

            for processor in self.processors:
                x_res = processor(x, edge_index)[0]
                x = x + F.relu(x_res)
            out = self.decoder(x)

        return out


# ! Try MoNet
# class SimulatorModel(torch.nn.Module):
#     def __init__(self, num_features, num_targets, num_layers, num_hidden):
#         # def __init__(self):
#         super(SimulatorModel, self).__init__()

#         self.num_features = num_features
#         self.num_targets = num_targets

#         # self.eb = EdgeModel(
#         #     nn.Sequential(
#         #         nn.Linear(5, num_hidden),
#         #         nn.ReLU(),
#         #         nn.Linear(num_hidden, num_hidden),
#         #         nn.ReLU(),
#         #         nn.Linear(num_hidden, 5),
#         #     )
#         # )
#         # self.nb = NodeModel(
#         #     nn.Sequential(
#         #         nn.Linear(self.num_features + 2, num_hidden),
#         #         nn.ReLU(),
#         #         nn.Linear(num_hidden, num_hidden),
#         #         nn.ReLU(),
#         #         nn.Linear(num_hidden, self.num_features),
#         #     )
#         # )

#         # self.processors = torch.nn.ModuleList()

#         # # # Shared parameter
#         # # for i in range(num_layers - 1):
#         # #     self.processors.append(self.gn)

#         # # Not Shared parameter
#         # for i in range(num_layers - 1):
#         #     self.processors.append(MetaLayer(edge_model=self.eb, node_model=self.nb))

#         # self.decoder = nn.Sequential(
#         #     nn.Linear(self.num_features, num_hidden),
#         #     nn.ReLU(True),
#         #     nn.Linear(num_hidden, num_hidden),
#         #     nn.ReLU(True),
#         #     nn.Linear(num_hidden, num_hidden),
#         #     nn.ReLU(True),
#         #     nn.Linear(num_hidden, self.num_targets),
#         # )

#         # ! MoNet
#         self.in_conv = GMMConv(
#             self.num_features, num_hidden, 3, 4, separate_gaussians=True
#         )
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.convs.append(
#                 GMMConv(num_hidden, num_hidden, 3, 4, separate_gaussians=True)
#             )

#         self.out_cov = GMMConv(
#             num_hidden, self.num_targets, 3, 4, separate_gaussians=True
#         )

#         self.decoder1 = nn.Sequential(
#             nn.Linear(num_hidden, num_hidden),
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
#         f_r3 = torch.cat((f_r, f_r, f_r), dim=-1)

#         norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))
#         grad_r = torch.mul(f_r3, disp_r)

#         disp_r = torch.div(disp_r, norm_r)
#         edge_attr = disp_r

#         return edge_attr

#     def forward(self, data, mode):
#         # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         x, edge_index = data.x, data.edge_index

#         # print(edge_attr.shape)
#         edge_attr = self._compute_edge_attr(x, edge_index)
#         x_in = x[:, -2:]

#         if mode == 1:

#             # ! GN + Decoder

#             # for processor in self.processors:
#             #     x_res = processor(x, edge_index)[0]
#             #     x = x + F.relu(x_res)
#             # out = self.decoder(x)

#             # ! New GN + Decoder

#             for processor in self.processors:
#                 x_res = processor(x_in, edge_index, edge_attr)[0]
#                 x = x + F.relu(x_res)
#             out = self.decoder(x)

#             # ! MoNet + Decoder

#             # x = self.in_conv(x, edge_index, edge_attr)
#             # x = F.relu(x)

#             # for conv in self.convs:
#             #     # x = conv(x, edge_index, edge_attr)
#             #     # x = F.relu(x)
#             #     x_res = conv(x, edge_index, edge_attr)
#             #     x = x + F.relu(x_res)
#             # out = self.out_cov(x, edge_index, edge_attr)

#         return out
