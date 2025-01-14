import torch
def edge_index_to_adjacency_matrix(edge_index, num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)  
      
    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上  
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  
    adjacency_matrix[edge_index[1], edge_index[0]] = 1  
      
    return adjacency_matrix