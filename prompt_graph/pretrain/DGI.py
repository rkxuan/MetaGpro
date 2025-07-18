from .base import PreTrain
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset, uniform
from torch.optim import Adam
import torch
from torch import nn
import time
from prompt_graph.utils import generate_corrupted_graph
from prompt_graph.data import load4node, load4graph, NodePretrain
import os
import numpy as np
import copy

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class DGI(PreTrain):
    def __init__(self, target_dim=-1, pca=False, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        
        self.pca = pca
        self.Designated_dim = target_dim
      
        self.disc = Discriminator(self.hid_dim).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.graph_data = self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)  
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay = 0.0)

    def load_graph_data(self):
        
        if self.pca is False:  # 不用pca降维
            if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'ogbn-arxiv','Actor', 'Texas', 'Wisconsin']:
                data, self.input_dim, _ = load4node(self.dataset_name)  # 需要先读入数据，参数为dataset_name，为str格式
                if self.input_dim != self.Designated_dim and self.Designated_dim !=-1: # 维度一样或者指定维度是-1的话说明没有指定维度，主要是为了兼容原版的写法
                    self.input_dim = self.Designated_dim
                    data.x = data.x[:, :self.input_dim]

        else:    #pca指定了降维维度
            self.input_dim = self.Designated_dim
            if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'ogbn-arxiv','Actor', 'Texas', 'Wisconsin']:
                data, _ , _ = load4node(self.dataset_name)
                _, _, V = torch.pca_lowrank(data.x, self.input_dim)
                data.x = torch.matmul(data.x, V[:, :self.input_dim])

        return data

    def pretrain_one_epoch(self):
        self.gnn.train()
        self.optimizer.zero_grad()
        device = self.device

        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora','Computers', 'Photo']:
            graph_original = self.graph_data
            graph_corrupted = copy.deepcopy(graph_original)
            idx_perm = np.random.permutation(graph_original.x.size(0))
            graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

            graph_original.to(device)
            graph_corrupted.to(device)

            pos_z = self.gnn(graph_original.x, graph_original.edge_index)
            neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)

            s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

            logits = self.disc(s, pos_z, neg_z)

            lbl_1 = torch.ones((pos_z.shape[0], 1))
            lbl_2 = torch.zeros((neg_z.shape[0], 1))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

            loss = self.loss_fn(logits, lbl)
            loss.backward()
            self.optimizer.step()

            accum_loss = float(loss.detach().cpu().item())            
        elif self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'ogbg-ppa', 'DD']:
            accum_loss = torch.tensor(0.0)
            for batch_id, batch_graph in enumerate(self.batch_dataloader):
                graph_original = batch_graph.to(device)
                graph_corrupted = copy.deepcopy(graph_original)
                idx_perm = np.random.permutation(graph_original.x.size(0))
                graph_corrupted.x = graph_original.x[idx_perm].to(self.device)

                graph_original.to(device)
                graph_corrupted.to(device)

                pos_z = self.gnn(graph_original.x, graph_original.edge_index)
                neg_z = self.gnn(graph_corrupted.x, graph_corrupted.edge_index)
        
                s = torch.sigmoid(torch.mean(pos_z, dim=0)).to(device)

                logits = self.disc(s, pos_z, neg_z)

                lbl_1 = torch.ones((pos_z.shape[0], 1))
                lbl_2 = torch.zeros((neg_z.shape[0], 1))
                lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

                loss = self.loss_fn(logits, lbl)
                loss.backward()
                self.optimizer.step()

                accum_loss += float(loss.detach().cpu().item())
          
            accum_loss = accum_loss/(batch_id+1)

        return accum_loss    
            


    def pretrain(self):
        train_loss_min = 1000000
        patience = 20
        cnt_wait = 0

        for epoch in range(1, self.epochs + 1):
            time0 = time.time()
            train_loss = self.pretrain_one_epoch()
            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, self.epochs , train_loss))

            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break


        folder_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}/{'DGI'}.{self.gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)

        self.gnn = self.gnn.to('cpu')
        torch.save(self.gnn.state_dict(), file_path)


        print("+++model saved in path:", file_path)
        return file_path