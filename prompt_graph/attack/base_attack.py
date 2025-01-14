import torch
from prompt_graph.model import GAT, GCN, GIN, GraphTransformer
from torch import nn, optim
from prompt_graph.data import load4node, load4graph, load4node_to_sparse, split_train_val_test, sample_dataset
#from prompt_graph.prompt import LightPrompt_token, Gprompt
from prompt_graph.model import GAT, GCN, GIN, GraphTransformer
from torch.nn import functional as F
from prompt_graph.utils import Gprompt_tuning_loss
from prompt_graph.data import load4node, load4graph
import numpy as np
from prompt_graph.pretrain import DGI, GraphCL, SimGRACE
from prompt_graph.utils import seed_everything, edge_index_to_adjacency_matrix
import warnings
import copy
import os
from prompt_graph.attack import PreTrain_task


class BaseAttack(torch.nn.Module):
    def __init__(self, attack_structure=True, attack_features=False, pretrain_type='GraphMAE', gnn_type='GCN', hid_dim = 128, gln=2, num_pretrain_epoch=500,
    pretrain_dataset='PubMed', target_dim=100, pca=False, target_dataset='Cora', labeled_each_class=100, device='auto',
    pre_train_model_file_path='/root/autodl-tmp/MetaGpro/pre_trained_model/', ll_constraint=False, *arg, **kwargs):
    
        super(BaseAttack, self).__init__()

        self.attack_structure = attack_structure
        self.attack_features = attack_features

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.pretrain_type = pretrain_type
        self.pretrain_gnn_type = gnn_type
        self.hid_dim = hid_dim
        self.num_layer = gln
        self.num_pretrain_epoch = num_pretrain_epoch

        self.pretrain_dataset = pretrain_dataset
        self.target_dataset = target_dataset

        self.labeled_each_class = labeled_each_class   # k-shot
        self.ll_constraint = ll_constraint

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        if pretrain_dataset == target_dataset:
            self.data, self.input_dim, self.out_dim = load4node_to_sparse(target_dataset)  # load as sparse tensor 
            self.target_dim = -1
            self.pca = pca
        else:
            _, pretrain_feature_dim, _ = load4node_to_sparse(pretrain_dataset)
            self.data, target_feature_dim, self.out_dim = load4node_to_sparse(target_dataset)
            self.target_dim = min(target_dim, pretrain_feature_dim, target_feature_dim)
            self.input_dim = target_dim
            self.pca = pca
            self.transform_feature_dim()

        if target_dataset in ['PubMed']:
            # PubMed is such a big dataset, so we sample 5k nodes
            self.data = sample_dataset(self.data, self.out_dim, self.labeled_each_class, 5000)
        elif target_dataset in ['Computers', 'Photo']:
            self.data = sample_dataset(self.data, self.out_dim, self.labeled_each_class, 3000)
 

        if target_dataset in ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo']:
            self.undirected = True
        else:           
            self.undirected = False
        
        # device has been defined in BaseAttack
        
        #split dataset
        self.train_mask, self.val_mask, self.test_mask = split_train_val_test(self.data, self.out_dim, self.labeled_each_class)

        # pretrain a model or load
        self.pre_train_model_path = os.path.join(pre_train_model_file_path, f"{self.pretrain_dataset}/{self.pretrain_type}.{self.pretrain_gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt")
        if not self.load_pretrain_model():
            self.pretrain_stage()

        # set weight for changes
        adj_dense = self.data.adj_t.to_dense()
        self.nedges = (adj_dense>0).nonzero().shape[0]    # nedge in original dataset
        self.adj_ori = adj_dense.to(self.device)
        self.nnodes  = adj_dense.shape[0]



        # dataset part
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
    
        self.y_self_training = None

        self.check_adj_tensor()

    def load_pretrain_model(self):
        # we can  try to load_model at first,if not exist,then pretrain a model.
        if self.pretrain_gnn_type == 'GAT':
                self.pretrain_gnn = GAT(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.pretrain_gnn_type == 'GCN':
                self.pretrain_gnn = GCN(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.pretrain_gnn_type == 'GIN':
                self.pretrain_gnn = GIN(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.pretrain_gnn_type == 'GraphTransformer':
                self.pretrain_gnn = GraphTransformer(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)

        #file_path = path + f"{self.pretrain_dataset}/{self.pretrain_type}.{self.pretrain_gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt"

        if not os.path.exists(self.pre_train_model_path):
            print("Pretrain model not exists,then we will pretrain a model as required")
            return 0
        
        self.pretrain_gnn.load_state_dict(torch.load(self.pre_train_model_path))
        self.pretrain_gnn.to(self.device)
        self.pretrain_gnn.eval()      # Frozen gnn model
        
        return 1

    def pretrain_stage(self):
        file_path = PreTrain_task(self.pretrain_type, self.pretrain_gnn_type, self.hid_dim, self.num_layer, 
        self.num_pretrain_epoch, self.pretrain_dataset, self.input_dim, self.pca).pre_train_stage()
        """     
        if file_path != self.pre_train_model_path:
            print(file_path)
            print(self.pre_train_model_path)
        else:
            print("work")"""
        

        self.pretrain_gnn.load_state_dict(torch.load(file_path))
        self.pretrain_gnn.to(self.device)
        self.pretrain_gnn.eval()      # Frozen gnn model



    def transform_feature_dim(self):
        if self.pca is False:  # 不用pca降维
            self.data.x = self.data.x[:, :self.input_dim]
        else:    #pca指定了降维维度
            _, _, V = torch.pca_lowrank(self.data.x, self.input_dim)
            self.data.x = torch.matmul(self.data.x, V[:, :self.input_dim])
        

    def check_adj_tensor(self):
        assert torch.abs(self.adj_ori - self.adj_ori.t()).sum() == 0, "Input graph is not symmetric"
        assert self.adj_ori.max() == 1, "Max value should be 1!"
        assert self.adj_ori.min() == 0, "Min value should be 0!"
        diag = self.adj_ori.diag()
        assert diag.sum() == self.adj_ori.shape[0], "when you load data, there should be Addselfloop"

    def attack(self):
        self.modified_adj = self.adj_ori.detach().to('cpu')
        self.modified_features = self.x.detach().to('cpu')


    

    

