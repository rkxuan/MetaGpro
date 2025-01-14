import torch
from prompt_graph.utils import seed_everything
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor


"""def sample_dataset(data, num_classes, nnodes_each, sample_nnodes):
    # nnodes_each is to make sure there are at least nnodes for each label in sample dataset

    x = data.x
    y = data.y
    adj_t = data.adj_t
    adj = adj_t.to_dense()

    #print("check:", num_classes, nnodes_each, sample_nnodes)

    nnodes = x.shape[0]
    for i in range(20):
        index = torch.randperm(nnodes)[:sample_nnodes]
        sample_y = y[index]
        judge = 1
        for j in range(num_classes):
            sample_y_j = torch.where(sample_y==j)[0]
            if sample_y_j.shape[0] < nnodes_each:
                judge = 0
                break
            
        if judge:
            sample_x = x[index]
            sample_adj = adj[index][:, index]
            sample_edge_index_0, sample_edge_index_1 = torch.where(sample_adj)
            sample_edge_index = torch.stack([sample_edge_index_0, sample_edge_index_1], dim=0)    
            data = Data(x=sample_x, y=sample_y, edge_index=sample_edge_index)
            sparse = ToSparseTensor()
            data = sparse(data)
            return data
        else:
            continue
    raise RuntimeError('Sampling a small dataset was not successfull. Please decrease `nnodes_each` or increase `sample_nnodes`.')"""

def sample_dataset(data, num_classes, nnodes_each, sample_nnodes):
    # nnodes_each is to make sure there are at least nnodes for each label in sample dataset

    x = data.x
    y = data.y
    adj_t = data.adj_t
    adj = adj_t.to_dense()

    mask = torch.zeros_like(y)

    for i in range(num_classes):
        i_index =  torch.where(y==i)[0]
        if i_index.shape[0] < nnodes_each:
            raise RuntimeError('Sampling a small dataset was not successfull. Please decrease `nnodes_each`.')
        sample_ = torch.randperm(i_index.shape[0])[:nnodes_each]
        sample_index = i_index[sample_]
        mask[sample_index] = 1

    remain_index = torch.where(mask==0)[0]
    sample_ = torch.randperm(remain_index.shape[0])[:(sample_nnodes-num_classes*nnodes_each)]
    sample_index = remain_index[sample_]
    mask[sample_index] = 1

    final_index = torch.where(mask==1)[0]

    sample_x = x[final_index]
    sample_y = y[final_index]
    sample_adj = adj[final_index][:, final_index]
    sample_edge_index_0, sample_edge_index_1 = torch.where(sample_adj)
    sample_edge_index = torch.stack([sample_edge_index_0, sample_edge_index_1], dim=0)    
    data = Data(x=sample_x, y=sample_y, edge_index=sample_edge_index)
    sparse = ToSparseTensor()
    data = sparse(data)
    return data
        
def Test():
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(), T.ToSparseTensor()])
    
    dataset = Planetoid(root='/root/autodl-tmp/Planetoid', name='Cora', transform=transform)
    data = dataset[0]

    sample_data = sample_dataset(data, dataset.num_classes, 100, 1000)
    print(sample_data)
    for i in range(dataset.num_classes):
        print(torch.where(sample_data.y==i)[0].shape[0])

if __name__ == '__main__':
    Test()