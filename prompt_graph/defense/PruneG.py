import torch.nn.functional as F
import torch

def prune(data, dropout=0.1):
    edge_index = data.edge_index

    x = data.x

    edge_similarities = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)

    _, sorted_indices = torch.sort(edge_similarities)

    num_elements = int(edge_index.size()[1] * dropout)

    mask_indices = sorted_indices[:num_elements]

    mask = torch.ones_like(edge_index[0], dtype=torch.bool)
    
    mask[mask_indices] = False

    pruned_edge_index = edge_index[:, mask]

    data.x = x

    data.edge_index = pruned_edge_index