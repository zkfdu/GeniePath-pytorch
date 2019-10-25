import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class simplify(MessagePassing):
    def __init__(self, requires_grad=True, **kwargs):
        super(simplify, self).__init__(aggr='add', **kwargs)

        self.requires_grad = requires_grad

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x_norm = F.normalize(x, p=2, dim=-1)

        return self.propagate(
            edge_index, x=x, x_norm=x_norm, num_nodes=x.size(0))

    def message(self, edge_index_i, x_j, x_norm_i, x_norm_j, num_nodes):
        # Compute attention coefficients.
        beta = self.beta if self.requires_grad else self._buffers['beta']
        alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        return x_j * alpha.view(-1, 1)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
