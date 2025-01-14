import torch
import numpy as np
from torch_geometric.utils import degree
import copy


class base_augmentation:
    def __init__(self, adj, features, aug_ratio=0.1, undirected=True):
        self.adj = adj
        self.features = features
        assert 0<=aug_ratio<1, "please set aug_ratio between 0 and 1"
        self.aug_ratio = 0.1
        self.undirected = undirected
    
    def augment(self):
        pass
    
    def get_name(self):
        pass


class edge_dropping(base_augmentation):
    def __init__(self, *arg, **kwargs):
        super(edge_dropping, self).__init__(*arg, **kwargs)

    def augment(self):
        # when loading data, we will do 'Addselfloop'
        diag = self.adj.diag()
        a_diag = torch.diag_embed(diag)
        adj_ = self.adj - a_diag

        edge_index_0, edge_index_1 = torch.where(adj_ == 1)
        nedge = edge_index_0.shape[0]//2 if self.undirected else edge_index_0.shape[0]
        edge_index =  torch.stack([edge_index_0, edge_index_1], dim=0)

        permute_num = int(nedge * self.aug_ratio)

        edge_index = edge_index.numpy()
        if self.undirected:
            up_index = np.where(edge_index[1]> edge_index[0])[0]
            edge_index = edge_index[:, up_index]

        idx_remained = np.random.choice(nedge, (nedge - permute_num), replace=False)
        edge_index = edge_index[:, idx_remained]

        adj_new = torch.zeros_like(self.adj)
        adj_new[edge_index[0,:], edge_index[1,:]] = 1
        if self.undirected:
            adj_new[edge_index[1,:], edge_index[0,:]] = 1

        adj_new = adj_new + a_diag

        return adj_new, self.features
    
    def get_name(self):
        return "edge_dropping"

class feature_masking(base_augmentation):
    def __init__(self, *arg, **kwargs):
        super(feature_masking, self).__init__(*arg, **kwargs)
    
    def augment(self):
        row_index, col_index = torch.where(self.features==1)
        feature_index = torch.stack([row_index, col_index], dim=0)
        n_features = feature_index.shape[1]

        permute_num = int(n_features*self.aug_ratio)
        feature_index = feature_index.numpy()
        idx_remained = np.random.choice(n_features, (n_features-permute_num), replace=False)

        feature_index = feature_index[:, idx_remained]

        feature_new = torch.zeros_like(self.features)
        feature_new[feature_index[0,:],feature_index[1,:]] = 1

        return self.adj, feature_new
    
    def get_name(self):
        return "feature_masking"


class edge_adding(base_augmentation):
    def __init__(self, *arg, **kwargs):
        super(edge_adding, self).__init__(*arg, **kwargs)
    
    def augment(self):
        # when loading data, we will do 'Addselfloop'
        diag = self.adj.diag()
        a_diag = torch.diag_embed(diag)
        adj_ = self.adj - a_diag

        edge_index_0, edge_index_1 = torch.where(adj_ == 1)
        edge_index_0_, edge_index_1_ = torch.where(self.adj == 0)
        nedge = edge_index_0.shape[0] // 2 if self.undirected else edge_index_0.shape[0]
        edge_index_ = torch.stack([edge_index_0_, edge_index_1_], dim=0)

        permute_num = int(nedge * self.aug_ratio)


        edge_index_ = edge_index_.numpy()
        if self.undirected:
            up_index = np.where(edge_index_[1] > edge_index_[0])[0]
            edge_index_ = edge_index_[:, up_index]

        nedge_ = edge_index_.shape[1]

        idx_choiced = np.random.choice(nedge_, permute_num, replace=False)
        edge_index_ = edge_index_[:, idx_choiced]
        #print(edge_index_)

        adj_new = torch.zeros_like(self.adj)
        adj_new[edge_index_[0, :], edge_index_[1, :]] = 1
        if self.undirected:
            adj_new[edge_index_[1, :], edge_index_[0, :]] = 1
        adj_new = self.adj + adj_new
        
        return adj_new, self.features

    def get_name(self):
        return "edge_adding"

class edge_weighted_dropping(base_augmentation):
    def __init__(self, *arg, **kwargs):
        super(edge_weighted_dropping, self).__init__(*arg, **kwargs)

    def augment(self):
        """
        Augmentation in Matacon-S:
        https://github.com/KanghoonYoon/torch-metacon/blob/main/src/graph/global_attack/metacon.py
        """
        diag = self.adj.diag()
        a_diag = torch.diag_embed(diag)
        adj_ = self.adj - a_diag

        edge_index_0, edge_index_1 = torch.where(adj_ == 1)
        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        deg = degree(edge_index[1])
        deg_col = deg[edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        weights = (s_col.max()-s_col) / (s_col.max()-s_col.mean())

        edge_weights = weights / weights.mean() * 0.5
        edge_weights = edge_weights.where(edge_weights < 0.7, torch.ones_like(edge_weights) * 0.7)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        edge_index_new = edge_index[:, sel_mask]

        adj_new = torch.zeros_like(self.adj)
        adj_new[edge_index_new[0, :], edge_index_new[1, :]] = 1
        adj_new = adj_new + a_diag
        
        return adj_new, self.features
    
    def get_name(self):
        return "edge_weighted_dropping"


class identity_augmentation(base_augmentation):
    def __init__(self, *arg, **kwargs):
        super(identity_augmentation, self).__init__(*arg, **kwargs)

    def augment(self):
        return self.adj, self.features
    
    def get_name(self):
        return "identity"