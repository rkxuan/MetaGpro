import torch
from prompt_graph.model import GCN, GIN, GAT, GraphTransformer
from prompt_graph.prompt import LightPrompt,HeavyPrompt, Gprompt
from prompt_graph.prompt import featureprompt, downprompt
from torch import nn, optim
from prompt_graph.utils import Gprompt_tuning_loss
from prompt_graph.data import load4node, load4graph
import numpy as np
from prompt_graph.pretrain import DGI, GraphCL, SimGRACE, GraphMAE, GInfoMinMax
from prompt_graph.utils import seed_everything, edge_index_to_adjacency_matrix
import warnings


class PreTrain_task:
    """
    为接下来的工作设置场景，需要完成的事情有：
    1、预训练好一个模型
    2、告知下游攻击任务,预训练模型接受的特征维度
    3、告知下游攻击任务,预训练模型接受的拓扑输入是边索引还是邻接矩阵(一般是边索引)
    """ 

    def __init__(self, pretrain_type='GraphCL',gnn_type='GCN', hid_dim = 128, gln=2, num_epoch=500,
    pretrain_dataset='Cora', target_dim=-1, pca=False):
        seed_everything(42)
        self.gnn_type = gnn_type
        self.pretrain_type = pretrain_type
        self.hid_dim = hid_dim
        self.gln = gln
        self.num_epoch = num_epoch
        self.pretrain_dataset = pretrain_dataset
        

        self.target_dim = target_dim
        self.pca = pca

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        #self.pre_train_stage()

    def pre_train_stage(self):
        if self.pretrain_type == 'GraphCL':
            self.pretrain_model = GraphCL(gnn_type=self.gnn_type, hid_dim=self.hid_dim, gln=self.gln, num_epoch=self.num_epoch, dataset_name=self.pretrain_dataset,target_dim=self.target_dim, pca=self.pca)
        
        elif self.pretrain_type == "SimGRACE":
            self.pretrain_model = SimGRACE(gnn_type=self.gnn_type, hid_dim=self.hid_dim,gln=self.gln,  num_epoch=self.num_epoch, dataset_name=self.pretrain_dataset, target_dim=self.target_dim, pca=self.pca)
        
        elif self.pretrain_type == "DGI":
            self.pretrain_model = DGI(gnn_type=self.gnn_type, hid_dim=self.hid_dim,gln=self.gln,  num_epoch=self.num_epoch, dataset_name=self.pretrain_dataset, target_dim=self.target_dim, pca=self.pca)
            #self.pretrain_model = DGI(gnn_type=self.gnn_type, dataset_name=self.pretrain_dataset)
        elif self.pretrain_type == "GraphMAE":
            self.pretrain_model = GraphMAE(gnn_type=self.gnn_type, hid_dim=self.hid_dim,gln=self.gln,  num_epoch=self.num_epoch, dataset_name=self.pretrain_dataset, target_dim=self.target_dim, pca=self.pca)
        elif self.pretrain_type == 'ADGCL':
            assert (self.gnn_type == 'GCN' or self.gnn_type == 'GIN'), "Only support GCN and GIN, attention in GAT or GT is somewhat constrast to edge_weight"
            self.pretrain_model = GInfoMinMax(gnn_type=self.gnn_type, hid_dim=self.hid_dim,gln=self.gln,  num_epoch=self.num_epoch, dataset_name=self.pretrain_dataset, target_dim=self.target_dim, pca=self.pca)
        else:
            raise ValueError(f"Unsupported pretrain type: {self.pretrain_type}")
        
        file_path = self.pretrain_model.pretrain()
        return file_path

    """
    def View_data(self, dataset_name):
        _, feature_dim, _ = load4node(dataset_name)
        return feature_dim
    """


def Test():
    #just for test
    PreTrain = PreTrain_task(pretrain_type='ADGCL', gnn_type='GCN', num_epoch=100)
    PreTrain.pre_train_stage()




if __name__ == '__main__':
    Test()
        



    