import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4node, load4graph, NodePretrain
from torch.optim import Adam
import os
from .base import PreTrain

class GraphCL(PreTrain):
    def __init__(self, target_dim=-1, pca=False, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.Designated_dim = target_dim
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)
    def load_graph_data(self):
        
        if self.pca is False:  # 不用pca降维
            if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo']:
                data, self.input_dim, _ = load4node(self.dataset_name)  # 需要先读入数据，参数为dataset_name，为str格式
                if self.input_dim != self.Designated_dim and self.Designated_dim !=-1: # 维度一样或者指定维度是-1的话说明没有指定维度，主要是为了兼容原版的写法
                    self.input_dim = self.Designated_dim
                    data.x = data.x[:, :self.input_dim]
                self.graph_list = NodePretrain(data = data, num_parts=200, split_method='Cluster')  #NodePretrain 没有dataname参数，此处应为pyG的data类型
                # self.graph_list, self.input_dim = NodePretrain(dataname = self.dataset_name, num_parts=0, split_method='Random Walk')
            
            else:
                self.input_dim, self.out_dim, self.graph_list= load4graph(self.dataset_name,pretrained=True)
                if self.input_dim != self.Designated_dim and self.Designated_dim !=-1: # 维度一样或者指定维度是-1的话说明没有指定维度
                    self.input_dim = self.Designated_dim
                    for graph in self.graph_list:
                        graph.x = graph.x[:, :self.input_dim]


        else:    #pca指定了降维维度
            self.input_dim = self.Designated_dim
            if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo']:
                data, _ , _ = load4node(self.dataset_name)
                _, _, V = torch.pca_lowrank(data.x, self.input_dim)
                data.x = torch.matmul(data.x, V[:, :self.input_dim])
                self.graph_list = NodePretrain(data = data, num_parts=200, split_method='Cluster')

            else:
                _, self.out_dim, self.graph_list= load4graph(self.dataset_name,pretrained=True)
                for graph in self.graph_list:
                    _, _, V = torch.pca_lowrank(graph.x, self.input_dim)
                    graph.x = torch.matmul(graph.x, V[:, :self.input_dim])

    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !

        return loader1, loader2
    
    def forward_cl(self, x, edge_index, batch, edge_weight=None):
        x = self.gnn(x, edge_index, edge_weight, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)  
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)  # 这函数写法有点高级 这里应该是余弦相似度
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]   # 按照这种写法 负例来自于除正例外的全部
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def train_graphcl(self, loader1, loader2, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001):

        epochs = self.epochs
        self.to(self.device)
        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            batch_size = 512
        loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'GraphCL', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        patience = 10
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):  # 1..100
            train_loss = self.train_graphcl(loader1, loader2, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, self.epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' epoch!')
                    break
            #print(cnt_wait)

        folder_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}/{'GraphCL'}.{self.gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
            
        self.gnn = self.gnn.to('cpu')
        torch.save(self.gnn.state_dict(), file_path)


        print("+++model saved in path:", file_path)
        return file_path
