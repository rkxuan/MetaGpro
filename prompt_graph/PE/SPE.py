from typing import Callable
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv
from .deepsets import DeepSets
from prompt_graph.model import 



class SPE(nn.Module):
    def __init__(self, PE_dims:int, hidden_dims:int=16, n_Deepsets:int=8) -> None:
        """
        原文中有两个参数
        phi对应排GNN编码器
        psi_list对应对Lambda排列不变的映射函数
        由于攻击并不需要追求极致的性能，对应关系为
        phi->一次聚合  psi_list->hidden_dims Deepsets_functions
        """
        self.Deepsets_list = nn.Modulelist([Deeepsets(n_layers=2,in_dims=1,hidden_dims=hidden_dims,out_dims=1) for i in range(n_Deepsets)])
        self.gcnconv = GCNConv(input_dims=n_Deepsets, out_dims=PE_dims)    # 像meta-attack追求速度，模块可以简单化
        
        super().__init__()
    
    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [D_pe]
        :param V: Concatenated eigenvector matrices. [N, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E]
        :return: Positional encoding matrix. [N, PE_dims]
        """

        Lambda = Lambda.unsqueeze(dim=1)                     # [D_pe, 1]
        
        Z = torch.stack([Deepsets(Lambda).squeeze(dim=1)     # [D_pe]
       for Deepsets in self.Deepsets_list], dim=1)           # [D_pe, M]

        
        V = V.unsqueeze(dim=0)                               # [1, N, D_pe]
        Z = Z.permute(1,0)                                   # [M, D_pe]
        Z = Z.diag_embed()                                   # [M, D_pe, D_pe]
        V_T = V.mT                                           # [1, D_pe, N]
        W = V.matmul(Z).matmul(V_T)                          # [M, N, N]
        

        W = W.permute(1, 2, 0)                               # [N, N, M]


        W = self.gcnconv(W, edge_index)                      # [N, N, PE_dims]

        W = W.mean(dim=1)                                    # [N, PE_dims]
        
        return W