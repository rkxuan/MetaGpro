import torch
from prompt_graph.attack import PreTrain_task
import torch.nn as nn
from prompt_graph.data import load4node, load4graph, load4node_to_sparse, split_train_val_test
# from prompt_graph.prompt import LightPrompt_token, Gprompt
from prompt_graph.model import GAT, GCN, GIN, GraphTransformer
from torch.nn import functional as F
from tqdm import tqdm
from prompt_graph.attack import BaseAttack
import os
import math
from torchmetrics import Accuracy
import copy
import numpy as np
import matplotlib.pyplot as plt
from prompt_graph.utils import likelihood_ratio_filter

import gc


class BaseMeta(BaseAttack):
    def __init__(self, *arg, **kwargs):
        super(BaseMeta, self).__init__(*arg, **kwargs)

        if self.attack_features:
            self.feature_changes = torch.nn.Parameter(torch.FloatTensor(self.nnodes, self.input_dim)).to(
                self.device)  # 可以看到如果要改变输入的话 输入也是要求梯度信息的
            self.feature_changes.data.fill_(0)

        if self.attack_structure:
            self.adj_changes = torch.nn.Parameter(torch.FloatTensor(self.nnodes, self.nnodes)).to(self.device)
            self.adj_changes.data.fill_(0)

    def get_modified_adj(self):
        # self.adj+changes是图扰动的变量，随后通过这个函数来修改邻接矩阵
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))  # 把对对角线的修改去掉
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)

        modified_adj = adj_changes_square + self.adj_ori  # 改变为原本的邻接矩阵+新的

        return modified_adj

    def get_modified_features(self):
        return self.x + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        大概意思就是说不希望原本孤立的节点被添加边
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio
    
    def get_adj_score(self, adj_grad, modified_adj):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask
        
        if self.ll_constraint:
            allowed_mask, _= self.log_likelihood_constraint(modified_adj, self.adj_ori, 0.04)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad


    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad

    def normalize_adj_tensor(self, adj):
        # adj_ = adj + torch.eye(adj.shape[0])      不用添加对角线元素， 因为加载数据的时候预处理过了
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -1 / 2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)

        adj_norm = D_mat_inv @ adj @ D_mat_inv  # GCN的归一化方式
        return adj_norm

    def save_adj(self, attack_name='mettack'):

        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'adj_mod'}.pt"

        modified_adj = self.get_modified_adj().cpu().clone()

        torch.save(modified_adj, os.path.join(self.save_fold_path, name))


    def save_feature(self, attack_name='mettack'):

        assert self.modified_features is not None, \
                'modified_adj is None! Please perturb the feature first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'feature_mod'}.pt"

        modified_features = self.get_modified_features().cpu().clone()

        torch.save(modified_features, os.path.join(self.save_fold_path, name))


class Metattack(BaseMeta):
    def __init__(self, train_iters:int=100, lr:float=0.1, momentum:float=0.9, lambda_:float=0.5, budget:float=0.05, save_fold_path="/root/autodl-tmp/deeprobust",
    attack_loss='CE', with_bias=False, with_relu=False, *arg, **kwargs):
        super(Metattack, self).__init__(*arg,**kwargs)
        

        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters  # better to set False empirically
        self.save_fold_path = save_fold_path
        self.attack_loss_name = attack_loss
        self.attack_loss_functions(attack_loss)
        self.with_bias = with_bias
        self.with_relu = with_relu     # 

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.train_acc_list = []
        self.val_acc_list = []


        assert 0<=lambda_<=1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_

        if self.undirected :
            self.n_perturbations = int(self.nedges * budget //2)
        else:
            self.n_perturbations = int(self.nedegs * budget)


        previous_size = self.input_dim
        hidden_sizes = [self.hid_dim for i in range(self.num_layer)]
        for ix, nhid in enumerate(hidden_sizes):
            weight = torch.nn.Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            w_velocity = torch.zeros(weight.shape).to(self.device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = torch.nn.Parameter(torch.FloatTensor(nhid).to(self.device))
                b_velocity = torch.zeros(bias.shape).to(self.device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = torch.nn.Parameter(torch.FloatTensor(previous_size, self.out_dim).to(self.device))
        output_w_velocity = torch.zeros(output_weight.shape).to(self.device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim).to(self.device))
            output_b_velocity = torch.zeros(output_bias.shape).to(self.device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def self_training(self, features, adj_norm):
        self._initialize()

        for ix in range(self.num_layer + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True

        optimizer = torch.optim.Adam(self.weights+self.biases, lr=self.lr)

        for j in tqdm(range(self.train_iters), desc="self-training"):
            optimizer.zero_grad()
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                hidden = adj_norm @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
            loss_labeled.backward()
            optimizer.step()

        
        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask==1] = self.y[self.train_mask==1]
  

    def inner_train(self, features, adj_norm):
        self._initialize()

        for ix in range(self.num_layer + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
        
                hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm):
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = self.attack_loss_func(output, self.y, self.train_mask==1)
        loss_unlabeled = self.attack_loss_func(output, self.y_self_training, self.train_mask==0)

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask==1], self.y[self.train_mask==1]).item()
        unlabeld_data_acc = accuracy(output[self.val_mask==1], self.y[self.val_mask==1]).item()

        self.train_acc_list.append(labeled_data_acc)
        self.val_acc_list.append(unlabeld_data_acc)

        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None


        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def attack(self):

        # self_training
        modified_features = self.x    
        modified_adj = self.adj_ori
        adj_norm = self.normalize_adj_tensor(modified_adj)
        self.self_training(modified_features, adj_norm)

        for i in tqdm(range(self.n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj()
            
            if self.attack_features:
                modified_features = self.get_modified_features()

            adj_norm = self.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_features, adj_norm)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)


            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = torch.div(adj_meta_argmax, self.nnodes, rounding_mode='trunc'), adj_meta_argmax % self.nnodes
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[col_idx][row_idx] + 1)
                #print("check:", torch.sum(self.adj_changes.data))
            
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = torch.div(feature_meta_argmax, self.input_dim, rounding_mode='trunc'), feature_meta_argmax % self.input_dim
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx]+1)
            #print(torch.cuda.memory_reserved()) 

        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None
        self.modified_features = self.get_modified_features().detach() if self.attack_features else None

    def attack_loss_functions(self, attack_loss):
        if attack_loss == 'CE':
            self.attack_loss_func = self.CE_loss
        elif attack_loss == 'GraD':
            self.attack_loss_func = self.GraD_loss
        elif attack_loss == 'Tanh':
            self.attack_loss_func = self.Tanh_loss
        elif attack_loss == 'Bias_Tanh':
            self.attack_loss_func = self.Bias_Tanh_loss
        elif attack_loss == 'MCE':
            self.attack_loss_func = self.MCE_loss
        else:
            raise ValueError(f"Unsupported attack loss type: {attack_loss}")

    def CE_loss(self, logits, labels, index):
        logits_ = logits[index]
        labels_ = labels[index]
        return F.nll_loss(logits_, labels_)

    def GraD_loss(self, logits, labels, index):
        return logits[index, labels[index]].mean()

    def Tanh_loss(self, logits, labels, index):
        sorted = logits.argsort(-1)
        best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
        margin = (
                logits[index, labels[index]]
                - logits[index, best_non_target_class[index]]
        )

        return torch.tanh(-margin).mean()

    def Bias_Tanh_loss(self, logits, labels, index, k=0.5):
        sorted = logits.argsort(-1)
        best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
        margin = (
                logits[index, labels[index]]
                - logits[index, best_non_target_class[index]]
        )

        loss1 = torch.tanh(-margin[margin >= 0])
        loss2 = k * torch.tanh(k * -margin[margin < 0])
        loss = torch.cat([loss1, loss2])

        return loss.mean()

    def MCE_loss(self, logits, labels, index):
        logits_, labels_ = logits[index], labels[index]
        not_flipped = logits_.argmax(-1) == labels_
        return -F.nll_loss(logits_[not_flipped], labels_[not_flipped])

    def plot_acc_during_training(self):
        x = np.arange(0, self.n_perturbations)
        train_acc = np.array(self.train_acc_list)
        val_acc = np.array(self.val_acc_list)

        plt.plot(x, train_acc, label="train_acc")
        plt.plot(x, val_acc, label="val_acc")

        plt.xlabel("N_perturbations")
        plt.ylabel("Accuracy(%)")
        plt.legend()

        if not os.path.exists("./result"):
            os.makedirs("./result")

        save_file_name = f"Mettack_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)

    
    def visualize_topology_attack_tendency(self):

        """
        原始数据太稀疏了 直接用余弦相似度基本上计算出来的就是0 
        x = self.x / (torch.norm(self.x, dim=-1, keepdim=True)+1e-8)
        similarity = torch.mm(x, x.T)    # (nnodes, nnodes)
        similarity = 0.5 * (similarity + 1)   # from (-1, 1) to (0, 1)
        因此还是用Jaccard相似度
        """

        x = self.x.data    
        x[x > 0] = 1
        x = x.int() 

        plt.xlabel("Jaccard similarity")
        plt.ylabel("Number of edge")

        index_0, index_1 = torch.where(self.adj_changes.data > 0)
        if index_0.shape != 0:
            add_jaccard = torch.zeros((index_0.shape[0]), device=x.device)
            # 计算每一行与其他行的Jaccard相似度
            for ix, index in enumerate(zip(index_0, index_1)):
                
                intersection = torch.sum((x[index[0]] & x[index[1]])).item()
                union = torch.sum((x[index[0]] | x[index[1]])).item()
                # 计算Jaccard相似度
                jaccard_index = intersection / union if union != 0 else 0
                add_jaccard[ix] = jaccard_index
            add_jaccard = add_jaccard.to('cpu').numpy()
            plt.hist(add_jaccard, bins=30, alpha=0.5, edgecolor='black', label='Add')


        index_0, index_1 = torch.where(self.adj_changes.data < 0)
        if index_0.shape != 0:
            delete_jaccard = torch.zeros((index_0.shape[0]), device=x.device)
            # 计算每一行与其他行的Jaccard相似度
            for ix, index in enumerate(zip(index_0, index_1)):
                
                intersection = torch.sum((x[index[0]] & x[index[1]])).item()
                union = torch.sum((x[index[0]] | x[index[1]])).item()
                # 计算Jaccard相似度
                jaccard_index = intersection / union if union != 0 else 0
                delete_jaccard[ix] = jaccard_index
            delete_jaccard = delete_jaccard.to('cpu').numpy()
            plt.hist(delete_jaccard, bins=30, alpha=0.5, color='red', edgecolor='black', label='Delete')
        
        
        plt.legend()
        save_file_name = f"Mettack_for_{self.target_dataset}_with_{self.n_perturbations}_topo_tendency.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)
                
    


class MetaApprox(BaseMeta):
    def __init__(self, train_iters: int = 100, lr: float = 0.1, momentum: float = 0.9, lambda_: float = 0.5,
                 budget: float = 0.05, save_fold_path="/root/autodl-tmp/deeprobust",
                 with_bias=False, with_relu=False, *arg, **kwargs):
        super(MetaApprox, self).__init__(*arg, **kwargs)

        self.momentum = momentum
        self.lr = lr
        # self.warming_up_iters = warning_up_iters
        self.train_iters = train_iters  # better to set False empirically
        self.save_fold_path = save_fold_path
        self.with_bias = with_bias
        self.with_relu = with_relu  #

        self.weights = []
        self.biases = []

        self.train_acc_list = []
        self.val_acc_list = []

        if self.attack_structure:
            self.adj_changes.retain_grad()
            self.adj_grad_sum = torch.zeros(self.nnodes, self.nnodes).to(self.device)
        if self.attack_features:
            self.feature_changes.retain_grad()
            self.feature_grad_sum = torch.zeros(self.nnodes, self.input_dim).to(self.device)

        assert 0 <= lambda_ <= 1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_

        if self.undirected:
            self.n_perturbations = int(self.nedges * budget // 2)
        else:
            self.n_perturbations = int(self.nedegs * budget)

        previous_size = self.input_dim
        hidden_sizes = [self.hid_dim for i in range(self.num_layer)]
        for ix, nhid in enumerate(hidden_sizes):
            weight = torch.nn.Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            self.weights.append(weight)

            if self.with_bias:
                bias = torch.nn.Parameter(torch.FloatTensor(nhid).to(self.device))
                self.biases.append(bias)

            previous_size = nhid

        output_weight = torch.nn.Parameter(torch.FloatTensor(previous_size, self.out_dim).to(self.device))
        self.weights.append(output_weight)

        if self.with_bias:
            output_bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim).to(self.device))
            self.biases.append(output_bias)
        self._initialize()

    def _initialize(self):
        # 建议和Mettack一样的初始化而不是deeprubost的写法，否则会出Nan， 我也不清楚为什么会出nan
        for w in self.weights:
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)


        if self.with_bias:
            for b in self.biases:
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)

        self.optimizer = torch.optim.Adam(self.weights + self.biases, lr=self.lr)

    def self_training(self, features, adj_norm):
        self._initialize()
    
        for j in tqdm(range(self.train_iters), desc="self-training"):
            self.optimizer.zero_grad()
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                hidden = adj_norm @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            self.optimizer.step()

        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.val_mask == 1], self.y[self.val_mask == 1]).item()
        print("you can use self-training to judge whether the hyper-parameter settings, such as lr, are reasonable or not")
        print("self-training acc on labeld_data:", labeled_data_acc)
        print("self-training acc on val_data:", unlabeld_data_acc)

        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask == 1] = self.y[self.train_mask == 1]

    def inner_train(self, features, modified_adj):
        adj_norm = self.normalize_adj_tensor(modified_adj)

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0

                hidden = adj_norm @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
            loss_unlabeled = F.nll_loss(output[self.train_mask==0], self.y_self_training[self.train_mask==0])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

            self.optimizer.step()

        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.val_mask == 1], self.y[self.val_mask == 1]).item()
        self.train_acc_list.append(labeled_data_acc)
        self.val_acc_list.append(unlabeld_data_acc)


    def attack(self):
        # self_training
        modified_features = self.x    
        modified_adj = self.adj_ori
        adj_norm = self.normalize_adj_tensor(modified_adj)
        self.self_training(modified_features, adj_norm)

        for i in tqdm(range(self.n_perturbations), desc="Perturbing graph"):
            self._initialize()

            if self.attack_structure:
                modified_adj = self.get_modified_adj()
                self.adj_grad_sum.data.fill_(0)

            if self.attack_features:
                modified_features = self.get_modified_features()
                self.feature_grad_sum.data.fill_(0)
            self.inner_train(modified_features, modified_adj)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = torch.div(adj_meta_argmax, self.nnodes,
                                             rounding_mode='trunc'), adj_meta_argmax % self.nnodes
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[col_idx][row_idx] + 1)
                # print("check:", torch.sum(self.adj_changes.data))

            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = torch.div(feature_meta_argmax, self.input_dim,
                                             rounding_mode='trunc'), feature_meta_argmax % self.input_dim
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)
            # print(torch.cuda.memory_reserved())

        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None
        self.modified_features = self.get_modified_features().detach() if self.attack_features else None


    def plot_acc_during_training(self):
        x = np.arange(0, self.n_perturbations)
        train_acc = np.array(self.train_acc_list)
        val_acc = np.array(self.val_acc_list)

        plt.plot(x, train_acc, label="train_acc")
        plt.plot(x, val_acc, label="val_acc")

        plt.xlabel("N_perturbations")
        plt.ylabel("Accuracy(%)")
        plt.legend()

        if not os.path.exists("./result"):
            os.makedirs("./result")

        save_file_name = f"Mettack_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)
    



def Test():
    model = Metattack(budget=0.05, target_dataset='PubMed')
    model.attack()
    model.plot_acc_during_training()


if __name__ == '__main__':
    Test()