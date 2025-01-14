import torch
from prompt_graph.attack import PreTrain_task
import torch.nn as nn
from prompt_graph.data import load4node, load4graph, load4node_to_sparse, split_train_val_test
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
from prompt_graph.utils import edge_adding, edge_dropping, feature_masking, edge_weighted_dropping
import random


class BaseMeta(BaseAttack):
    def __init__(self, *arg, **kwargs):
        super(BaseMeta, self).__init__(*arg, **kwargs)

        if self.attack_features:
            self.feature_changes = torch.nn.Parameter(torch.FloatTensor(self.nnodes, self.input_dim)).to(self.device)     # 可以看到如果要改变输入的话 输入也是要求梯度信息的
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

        modified_adj = adj_changes_square + self.adj_ori   #改变为原本的邻接矩阵+新的
    
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
    

    """
    def self_training_label(self, output):
        # 这个可以改, 比如过滤低置信度的
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train==1] = self.y[idx_train==1]
        return labels_self_training
    """

        
    def get_adj_score(self, adj_grad, modified_adj):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad = adj_meta_grad - adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad = adj_meta_grad - torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask

    
        return adj_meta_grad
    

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad

    def normalize_adj_tensor(self, adj):
        #adj_ = adj + torch.eye(adj.shape[0])      不用添加对角线元素， 因为加载数据的时候预处理过了
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -1/2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)

        adj_norm = D_mat_inv @ adj @ D_mat_inv   # GCN的归一化方式
        return adj_norm

    
        
"""
'Debiased Graph Poisoning Attack via Contrastive Surrogate Objective'
CIKM 24
"""


class Metacon_s(BaseMeta):
    def __init__(self, train_iters:int=100, lr:float=0.1, momentum:float=0.9, lambda_:float=0.5, beta_:float=0.1, budget:float=0.05, save_fold_path="/root/autodl-tmp/deeprobust",
    with_bias=False, with_relu=False, aug_ratio=0.2, augmentation="random", *arg, **kwargs):
        super(Metacon_s, self).__init__(*arg,**kwargs)
        

        self.momentum = momentum
        self.lr = lr
        #self.warming_up_iters = warning_up_iters
        self.train_iters = train_iters  # better to set False empirically
        self.save_fold_path = save_fold_path
        self.with_bias = with_bias
        self.with_relu = with_relu      # better to set False empirically

        self.aug_ratio = aug_ratio

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.train_acc_list = []
        self.val_acc_list = []


        assert 0<=lambda_<=1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_
        assert 0<=beta_<= 1, "beta_ should between 0 and 1"
        self.beta_ = beta_

        if self.undirected :
            self.n_perturbations = int(self.nedges * budget //2)

        augmentation_dict = {'edge_dropping':edge_dropping, 'edge_adding':edge_adding, 'feature_masking':feature_masking, "edge_weighted_dropping":edge_weighted_dropping}
        if augmentation not in augmentation_dict:
            augmentation = random.choice(list(augmentation_dict.keys()))
            self.augmentation_func = augmentation_dict[augmentation]
        else:
            self.augmentation_func = augmentation_dict[augmentation]

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
            output_bias = torch.nn.Parameter(torch.FloatTensor(self.nclass).to(self.device))
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
            optimizer.step()

            
        
        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask==1] = self.y[self.train_mask==1]
  

    def inner_train(self, features, adj_norm, modified_adj):
        self._initialize()

        adj_2, features_2 = self.augmented_view(features, modified_adj)
        adj_2_norm = self.normalize_adj_tensor(adj_2)

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

            hidden1 = features
            hidden2 = features_2
            
            for ix, w in enumerate(self.weights):

                b = self.biases[ix] if self.with_bias else 0
        
                hidden1 = adj_norm @ hidden1 @ w + b
                hidden2 = adj_2_norm @ hidden2 @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden1 = F.relu(hidden1)
                    hidden2 = F.relu(hidden2)

            output = F.log_softmax(hidden, dim=1)
            output1 = F.elu(hidden1)
            output2 = F.elu(hidden2)

            self.tau = 0.5
            f = lambda x: torch.exp(x/ self.tau)

            refl = f(torch.mm(output1, output1.t()))
            between = f(torch.mm(output1, output2.t()))

            loss_unlabeled = -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))
            loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
            loss = loss_labeled + self.beta_ * loss_unlabeled[self.train_mask==0].mean()

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
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

        loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
        loss_unlabeled = F.nll_loss(output[self.train_mask==0], self.y_self_training[self.train_mask==0])
        loss_val = F.nll_loss(output[self.val_mask==1], self.y[self.val_mask==1])

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

            self.inner_train(modified_features, adj_norm, modified_adj)

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

            torch.cuda.empty_cache()
            #print(torch.cuda.memory_reserved())
        
        """folder_path = self.save_fold_path + f"/mettack/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_path = folder_path + "/modified_adj.pt"
        if os.path.exists(file_path):
            os.remove(file_path)

        self.modified_adj = self.get_modified_adj().detach()
        torch.save(self.modified_adj, file_path)"""
        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None

        
        """folder_path = self.save_fold_path + f"/mettack/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_path = folder_path + "/modified_features.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
            
        self.modified_features = self.get_modified_features().detach()
        torch.save(self.modified_features, file_path)"""
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

        save_file_name = f"Metacon_s_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)

    def augmented_view(self, features, adj):
      
        adj_ori_ = adj.data.to('cpu')
        x_ = features.data.to('cpu')
        
        augmentation = self.augmentation_func(adj_ori_, x_, self.aug_ratio, self.undirected)
        adj_ori_2, x_2 = augmentation.augment()
        adj_ori_2, x_2 = adj_ori_2.to(self.device), x_2.to(self.device) 

        
        return adj_ori_2, x_2
    
    def save_adj(self, attack_name='metacon'):

        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'adj_mod'}.pt"

        modified_adj = self.get_modified_adj().cpu().clone()

        torch.save(modified_adj, os.path.join(self.save_fold_path, name))


    def save_feature(self, attack_name='metacon'):

        assert self.modified_features is not None, \
                'modified_adj is None! Please perturb the feature first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'feature_mod'}.pt"

        modified_features = self.get_modified_features().cpu().clone()

        torch.save(modified_features, os.path.join(self.save_fold_path, name))

class Metacon_d(BaseMeta):
    def __init__(self, train_iters:int=100, lr:float=0.1, momentum:float=0.9, lambda_:float=0.5, vic_coef1=1.0, vic_coef2=1.0, vic_coef3=0.04,
    beta_:float=0.1, budget:float=0.05, save_fold_path="/root/autodl-tmp/deeprobust",
    with_bias=False, with_relu=False, aug_ratio=0.2, augmentation="random", *arg, **kwargs):
        super(Metacon_d, self).__init__(*arg,**kwargs)
        

        self.momentum = momentum
        self.lr = lr
        #self.warming_up_iters = warning_up_iters
        self.train_iters = train_iters  # better to set False empirically
        self.save_fold_path = save_fold_path
        self.with_bias = with_bias
        self.with_relu = with_relu      # better to set False empirically

        self.aug_ratio = aug_ratio

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.train_acc_list = []
        self.val_acc_list = []


        assert 0<=lambda_<=1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_
        assert 0<=beta_<= 1, "beta_ should between 0 and 1"
        self.beta_ = beta_
        assert 0<=vic_coef1<=1, "vic_coef1 should between 0 and 1"
        self.vic_coef1 = vic_coef1
        assert 0<=vic_coef2<=1, "vic_coef1 should between 0 and 1"
        self.vic_coef2 = vic_coef2
        assert 0<=vic_coef3<=1, "vic_coef1 should between 0 and 1"
        self.vic_coef3 = vic_coef3


        if self.undirected :
            self.n_perturbations = int(self.nedges * budget //2)

        augmentation_dict = {'edge_dropping':edge_dropping, 'edge_adding':edge_adding, 'feature_masking':feature_masking, "edge_weighted_dropping":edge_weighted_dropping}
        if augmentation not in augmentation_dict:
            augmentation = random.choice(list(augmentation_dict.keys()))
            self.augmentation_func = augmentation_dict[augmentation]
        else:
            self.augmentation_func = augmentation_dict[augmentation]

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
            output_bias = torch.nn.Parameter(torch.FloatTensor(self.nclass).to(self.device))
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
            optimizer.step()

            
        
        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask==1] = self.y[self.train_mask==1]
  

    def inner_train(self, features, adj_norm, modified_adj):
        self._initialize()

        adj_2, features_2 = self.augmented_view(features, modified_adj)
        adj_2_norm = self.normalize_adj_tensor(adj_2)

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

            hidden1 = features
            hidden2 = features_2
            
            for ix, w in enumerate(self.weights):

                b = self.biases[ix] if self.with_bias else 0
        
                hidden1 = adj_norm @ hidden1 @ w + b
                hidden2 = adj_2_norm @ hidden2 @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden1 = F.relu(hidden1)
                    hidden2 = F.relu(hidden2)

            output = F.log_softmax(hidden, dim=1)

            # VIC REG
            z1 = F.elu(hidden1)
            z2 = F.elu(hidden2)

            bs, num_feat = z1.shape
            z1 = z1[self.train_mask==0]
            z2 = z2[self.train_mask==0]


            # mse loss
            mse_loss = F.mse_loss(z1, z2)

            # std loss
            z1 = z1 - z1.mean(dim=0)
            z2 = z2 - z2.mean(dim=0)
            std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4) 
            std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4) 
            std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
            
            # cov loss
            cov_z1 = (z1.T @ z1) / (bs - 1)
            cov_z2 = (z2.T @ z2) / (bs - 1)
            cov_loss = self.off_diagonal(cov_z1).pow_(2).sum().div(num_feat) + self.off_diagonal(cov_z2).pow_(2).sum().div(num_feat)

            loss_unlabeled = self.vic_coef1 * mse_loss + self.vic_coef2 * std_loss + self.vic_coef3 * cov_loss


            loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
            loss = loss_labeled + self.beta_ * loss_unlabeled

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
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

        loss_labeled = F.nll_loss(output[self.train_mask==1], self.y[self.train_mask==1])
        loss_unlabeled = F.nll_loss(output[self.train_mask==0], self.y_self_training[self.train_mask==0])
        loss_val = F.nll_loss(output[self.val_mask==1], self.y[self.val_mask==1])

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

            self.inner_train(modified_features, adj_norm, modified_adj)

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
        """folder_path = self.save_fold_path + f"/mettack/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_path = folder_path + "/modified_adj.pt"
        if os.path.exists(file_path):
            os.remove(file_path)

        self.modified_adj = self.get_modified_adj().detach()
        torch.save(self.modified_adj, file_path)"""
        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None

        
        """folder_path = self.save_fold_path + f"/mettack/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_path = folder_path + "/modified_features.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
            
        self.modified_features = self.get_modified_features().detach()
        torch.save(self.modified_features, file_path)"""
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

        save_file_name = f"Metacon_d_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)

    def augmented_view(self, features, adj):
        
        
        adj_ori_ = copy.deepcopy(adj.data.to('cpu'))
        x_ = copy.deepcopy(features.data.to('cpu'))

        augmentation = self.augmentation_func(adj_ori_, x_, self.aug_ratio, self.undirected)
        adj_ori_2, x_2 = augmentation.augment()
        adj_ori_2, x_2 = adj_ori_2.to(self.device), x_2.to(self.device) 

        #print(self.augmentation.get_name()," has done")
        
        return adj_ori_2, x_2

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



def Test():  
    model = Metacon_s(train_iters = 90, budget=0.05, augmentation="edge_weighted_dropping",aug_ratio=0.2)
    model.attack()
    model.plot_acc_during_training()

if __name__ == '__main__':
    Test()