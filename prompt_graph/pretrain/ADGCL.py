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
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter
from prompt_graph.model import GCN, GIN, GAT, GraphTransformer
from prompt_graph.pretrain import DGI, GraphCL, SimGRACE, GraphMAE



class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, encoder_out_dim=64, mlp_edge_model_dim=64):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = encoder_out_dim

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        )  # 论文中的mlp
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):

        node_emb = self.encoder(x, edge_index)
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)   #理论上对于无向图 (i,j)=(j,i) 但我懒得花功夫在强制图的无向性上了 数据增强也不强制要求这个
        edge_logits = self.mlp_edge_model(edge_emb)
        #edge_mask = (src == dst)  # when loaddata will add self-loop, so make sure self-loop well not delete
        return edge_logits


class GCL(torch.nn.Module):
    def __init__(self, encoder, out_dim, proj_hidden_dim=300):
        super(GCL, self).__init__()

        self.encoder = encoder

        self.proj_head = Sequential(Linear(out_dim, proj_hidden_dim), ReLU(inplace=True),
                                    Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None):

        z = self.encoder(x, edge_index, edge_weight=edge_weight)

        z = self.proj_head(z)
        # z shape -> Batch x proj_hidden_dim
        return z

    @staticmethod
    def calc_loss(x1, x2):
        # x and x_aug shape -> Batch x proj_hidden_dim
        T=0.2
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)  # 这是l2正则
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)  # 这函数写法有点高级 这里应该是余弦相似度
        sim_matrix = torch.exp(sim_matrix/T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]  # 按照这种写法 负例来自于除正例外的全部
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        #print(loss)
        return loss





class GInfoMinMax(PreTrain):
    # if loss=Nan, please try following solution:
    # (1) increase the reg_lambda  (2) increase the T in 81
    def __init__(self, target_dim=-1, pca=False, reg_lambda=5, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.pca = pca
        self.Designated_dim = target_dim
        self.reg_lambda = reg_lambda
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.initialize_viewlearner()
        self.model = GCL(self.gnn, self.hid_dim, self.hid_dim).to(self.device)


    def initialize_viewlearner(self):
        if self.gnn_type == 'GAT':
                gnn = GAT(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCN':
                gnn = GCN(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GIN':
                gnn = GIN(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
                gnn = GraphTransformer(input_dim = self.input_dim, hid_dim = self.hid_dim, num_layer = self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.viewlearner = ViewLearner(gnn, self.hid_dim, self.hid_dim)
        
        self.viewlearner.to(self.device)


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


    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        

        view_list = []
        for g in graph_list:
            view_g = Data(x=g.x, edge_index=g.edge_index)
            view_list.append(view_g)
  
        loader = DataLoader(view_list, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !

        return loader

    def train_adgraphcl(self, loader, view_optimizer, model_optimizer):
        train_loss_accum = 0
        view_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(loader):
            self.viewlearner.train()
            self.viewlearner.zero_grad()
            self.model.eval()

            batch =  batch.to(self.device)

            x = self.model(batch.x, batch.edge_index, None)

            edge_logits = self.viewlearner(batch.x, batch.edge_index)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug = self.model(batch.x, batch.edge_index, batch_aug_edge_weight)

            # regularization

            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(self.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            view_loss = self.model.calc_loss(x, x_aug) - (self.reg_lambda * reg)
        
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()


            # train (model) to minimize contrastive loss
            self.model.train()
            self.viewlearner.eval()
            self.model.zero_grad()

            x = self.model(batch.x, batch.edge_index)
            edge_logits = self.viewlearner(batch.x, batch.edge_index)


            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug = self.model(batch.x, batch.edge_index, batch_aug_edge_weight)

            model_loss = self.model.calc_loss(x, x_aug)
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()


            train_loss_accum += float(model_loss.detach().cpu().item())
            view_loss_accum += float(view_loss.detach().cpu().item())
            total_step = total_step + 1
        return train_loss_accum / total_step, view_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.001, decay=0.0001):
        
        epochs = self.epochs
        self.batch_size = batch_size

        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            self.batch_size = 512
        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'ADGCL', self.gnn_type))
        view_optimizer = Adam(self.viewlearner.parameters(), lr=lr, weight_decay=self.weight_decay)
        model_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        patience = 20
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):  # 1..100
            train_loss, view_loss = self.train_adgraphcl(loader, view_optimizer, model_optimizer)

            print("***epoch: {}/{} | train_loss: {:.8} | view_loss: {:.8}".format(epoch, self.epochs, train_loss, view_loss))

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

        file_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}/{'ADGCL'}.{self.gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
            
        self.gnn = self.gnn.to('cpu')
        torch.save(self.gnn.state_dict(), file_path)


        print("+++model saved in path:", file_path)
        return file_path
