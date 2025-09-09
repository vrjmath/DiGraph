import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class DenseGraphConvolution(nn.Module):
    """Graph convolution using adjacency matrix multiplication (like old project)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: [num_nodes, in_features]
        adj: [num_nodes, num_nodes] adjacency matrix (dense or sparse)
        """
        support = x @ self.weight  # same as torch.mm
        output = adj @ support     # same as torch.spmm
        if self.bias is not None:
            output = output + self.bias
        return output

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.gc1 = DenseGraphConvolution(in_dim, hidden_dim)
        self.gc2 = DenseGraphConvolution(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj, batch):
        """
        x: node embeddings [num_nodes, in_dim]
        adj: adjacency matrix [num_nodes, num_nodes] (dense)
        batch: batch vector for pooling [num_nodes]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = global_mean_pool(x, batch)
        return self.fc(x)
