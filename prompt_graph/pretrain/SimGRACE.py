import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from prompt_graph.utils import mkdir
from torch.optim import Adam
from prompt_graph.data import load4node, load4graph, NodePretrain
from copy import deepcopy
from.base import PreTrain
import os

class SimGRACE(PreTrain):

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
        
    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
        return loader
    
    def forward_cl(self, x, edge_index, batch, edge_weight=None):
        x = self.gnn(x, edge_index, edge_weight, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def perturbate_gnn(self, data):
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2
    
    def train_simgrace(self, loader, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data) 
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1
            # print("第{}次gnn batch传播，loss是{}".format(step,train_loss_accum))

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.01,decay=0.0001):
        epochs = self.epochs
        if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
            batch_size = 512
        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        patience = 10
        cnt_wait = 0
        for epoch in range(1, self.epochs + 1):  # 1..100

            train_loss = self.train_simgrace(loader, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, self.epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break
            #print(cnt_wait)

        folder_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = f"/root/autodl-tmp/MetaGpro/pre_trained_model/{self.dataset_name}/{'SimGRACE'}.{self.gnn_type}.{str(self.num_layer)+'layers'}.{str(self.hid_dim)+'hidden_dim'}.{str(self.input_dim)+'input_dim'}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)

        self.gnn = self.gnn.to('cpu')
        torch.save(self.gnn.state_dict(), file_path)


        print("+++model saved in path:", file_path)
        return file_path