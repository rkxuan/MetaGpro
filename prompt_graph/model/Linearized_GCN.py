import torch
import torch.nn
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from prompt_graph.data import load4node, load4graph
from prompt_graph.utils import edge_index_to_adjacency_matrix, act
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention, global_mean_pool
from torch_geometric.nn.inits import glorot

# 使用方式见Test()

class GCNConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))       #这构建了一个要求梯度的 参数矩阵
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))               #偏置项
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)                           
        if self.bias is not None:                       
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Linearized_GCN(torch.nn.Module):    
    def __init__(self, input_dim, hid_dim, out_dim=None, num_layer=2, bias=False):
        super().__init__()

        GraphConv = GCNConvolution
        
        assert 1<num_layer, "please set num_layer>1"

        if out_dim is None:
            out_dim = hid_dim
        
        if num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim, bias), GraphConv(hid_dim, out_dim, bias)])
        else:
            layers = [GraphConv(input_dim, hid_dim, bias)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim, bias))
            layers.append(GraphConv(hid_dim, out_dim, bias))
            self.conv_layers = torch.nn.ModuleList(layers)

    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()

    def forward(self, x, adj_norm):
        for conv in self.conv_layers:
            x = conv(x, adj_norm)
        return x
    

def Test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, feature_dim, out_dim = load4node('Cora')
    x = data.x
    y = data.y
    edge_index = data.edge_index

    x = torch.FloatTensor(data.x).to(device)
    y = y.to(device)

    adj = edge_index_to_adjacency_matrix(edge_index, x.shape[0])

    adj_ = adj + torch.eye(adj.shape[0])
    D = torch.sum(adj_, dim=1)
    D_inv = torch.pow(D, -1/2)
    D_inv[torch.isinf(D_inv)] = 0.
    D_mat_inv = torch.diag(D_inv)

    adj_norm = D_mat_inv @ adj_ @ D_mat_inv   # GCN的归一化方式
    adj_norm = adj_norm.to(device)

    model = Linearized_GCN(feature_dim, hid_dim=128, out_dim=out_dim).to(device)
    #print(model.conv_layers[0].weight, model.conv_layers[1].weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

    for i in range(100):
        model.train()
        optimizer.zero_grad()
        output = F.log_softmax(model(x, adj_norm))
        loss = F.nll_loss(output, y)
        loss.backward()
        print("loss is:", loss)
        optimizer.step()
    

if __name__ == '__main__':
    Test()