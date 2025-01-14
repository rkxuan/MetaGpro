import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from prompt_graph.utils import act
from sklearn.cluster import KMeans
from torch_geometric.nn.inits import glorot


class SmoothPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num):
        super(SmoothPrompt, self).__init__()

        self.tokens = torch.nn.Parameter(torch.empty(token_num, token_dim))

        self.token_init()

        self.token_num = token_num

    def token_init(self):
        glorot(self.tokens)  #这是一个初始化

    def add(self, x:torch.Tensor):
        weight = torch.mm(x, torch.transpose(self.tokens, 0, 1))     # (n_nodes, token_nums)
        weight = torch.softmax(weight, dim=1)
        weighted_prompt_tokens = torch.mm(weight, self.tokens)    # (n_nodes, input_dim)

        return x  + weighted_prompt_tokens



class SparsePrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num, threshold:float=0.5):
        
        super(SparsePrompt, self).__init__()
    
        assert 0.4<=threshold<=1, "In Sparse-All-in-one, default 0.4<= threshold <=1"

        self.threshold = threshold

        self.tokens = torch.nn.Parameter(torch.empty(token_num, token_dim))

        self.token_init()

        self.token_num = token_num

    def token_init(self):

        glorot(self.tokens)  #这是一个初始化

    def forward(self, graph_batch: Batch):   
        # 如同论文中描述的
        # input  (n_nodes, input_dim) and input_dim == token_dim

        """
        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            weight = torch.mm(g.x, torch.transpose(self.tokens, 0, 1))     # (n_nodes, token_nums)
            weight = torch.sigmoid(weight)
            mask = weight < threshold
            masked_weight = weight.masked_fill(mask, 0)                      # (n_nodes, token_nums)

            weighted_prompt_tokens = torch.mm(masked_weight, self.tokens)    # (n_nodes, input_dim)
            x = g.x + 1/self.token_num * weighted_prompt_tokens

            data = Data(x=x, edge_index=g.edge_index, y=g.y)
            re_graph_list.append(data)
        
        graphp_batch = Batch.from_data_list(re_graph_list)
        """
        
        weight = torch.mm(graph_batch.x, torch.transpose(self.tokens, 0, 1))     # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.threshold
        masked_weight = weight.masked_fill(mask, 0)                      # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.tokens)    # (n_nodes, input_dim)
        graph_batch.x = graph_batch.x + 1/self.token_num * weighted_prompt_tokens
        return graph_batch
    

    def Tune(self, train_loader, gnn, answering, lossfn, opi, device, verbose=False):
    # 调用这个函数的时候 会轮流对answering和Prompt的参数进行优化
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader): 
            opi.zero_grad() 
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            # print(prompted_graph)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
            if verbose:
                print(' batch {}/{} | loss: {:.8f}'.format( batch_id, len(train_loader), train_loss))
        return running_loss / len(train_loader)


    def add(self, x: torch.Tensor):
        weight = torch.mm(x, torch.transpose(self.tokens, 0, 1))     # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.threshold
        masked_weight = weight.masked_fill(mask, 0)                      # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.tokens)    # (n_nodes, input_dim)

        return x + 1/self.token_num * weighted_prompt_tokens


class AllinonePrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num, threshold:float=0.2):
        
        super(AllinonePrompt, self).__init__()
    
        assert 0.0<=threshold<=1, "In Sparse-All-in-one, default 0.0<= threshold <=1"

        self.threshold = threshold

        self.tokens = torch.nn.Parameter(torch.empty(token_num, token_dim))

        self.token_init()

    def token_init(self):

        glorot(self.tokens)  #这是一个初始化

    def add(self, x: torch.Tensor):
        weight = torch.mm(x, torch.transpose(self.tokens, 0, 1))     # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.threshold
        masked_weight = weight.masked_fill(mask, 0)                      # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.tokens)    # (n_nodes, input_dim)

        return x + weighted_prompt_tokens

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num  # 这里首先先把prompt的索引放在前面；原来图的索引移向后面
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)  #随后这里计算prompt与原图间的相似度作为边
            # 非常值得注意的是 这样在一阶消息传递的时候 表征就等价于论文中的形式
            

            # 这两行是建立prompt图与原图间的边索引
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            # 添加prompt的表征，注意是在前
            x = torch.cat([pg.x, g.x], dim=0)    
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1) #现在是三部分  原来图的拓扑 提示图的拓扑以及原来-提示图的拓扑
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
    

    def Tune(self, train_loader, gnn, answering, lossfn, opi, device, verbose=False):
    # 调用这个函数的时候 会对answering和Prompt的参数进行优化
    # 但实际实现的时候 是一个轮流的过程 在node_task主函数中 会交替 eval prompt train answering和反过程
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader): 
            opi.zero_grad() 
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()
            if verbose:
                print(' batch {}/{} | loss: {:.8f}'.format( batch_id, len(train_loader), train_loss))
        return running_loss / len(train_loader)
    
    def TuneWithoutAnswering(self, train_loader, gnn, answering, lossfn, opi, device):  #是论文中without task head的实现
        total_loss = 0.0 
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            emb0 = gnn(batch.x, batch.edge_index, batch.batch)
            pg_batch = self.inner_structure_update()
            pg_batch = pg_batch.to(self.device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            sim = torch.softmax(dot, dim=1)
            loss = lossfn(sim, batch.y)     # 对于每个prompt来说对应于一个y
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()  
        return total_loss / len(train_loader) 

class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pres