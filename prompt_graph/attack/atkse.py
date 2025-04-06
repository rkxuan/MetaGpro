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
        # adj_ = adj + torch.eye(adj.shape[0])      不用添加对角线元素， 因为加载数据的时候预处理过了
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -1 / 2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)

        adj_norm = D_mat_inv @ adj @ D_mat_inv  # GCN的归一化方式
        return adj_norm

    def save_adj(self, attack_name='GraD'):

        assert self.modified_adj is not None, \
            'modified_adj is None! Please perturb the graph first.'

        name = f"{self.target_dataset}/{attack_name}.{self.budget + 'budget'}.{'adj_mod'}.pt"

        modified_adj = self.get_modified_adj().cpu().clone()

        torch.save(modified_adj, os.path.join(self.save_fold_path, name))

    def save_feature(self, attack_name='GraD'):

        assert self.modified_features is not None, \
            'modified_adj is None! Please perturb the feature first.'

        name = f"{self.target_dataset}/{attack_name}.{self.budget + 'budget'}.{'feature_mod'}.pt"

        modified_features = self.get_modified_features().cpu().clone()

        torch.save(modified_features, os.path.join(self.save_fold_path, name))


class AtkSE(BaseMeta):
    def __init__(self, train_iters: int = 100, lr: float = 0.1, momentum: float = 0.9, lambda_: float = 0.5,
                 budget: float = 0.05, save_fold_path="/root/autodl-tmp/deeprobust",
                 with_bias=False, with_relu=False, *arg, **kwargs):
        super(AtkSE, self).__init__(*arg, **kwargs)

        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        self.momentum_grad = None
        self.mom = 0.7

        self.dropnode = 0.05
        self.gauss_noise = 2e-4
        self.smooth_loop = 40
        self.wait_list = 4
        self.intervals = 0.25
        self.candidates = 32

        self.with_bias = with_bias
        self.with_relu = with_relu


        if self.undirected :
            self.n_perturbations = int(self.nedges * budget //2)
        else:
            self.n_perturbations = int(self.nedegs * budget)

        previous_size = self.input_dim
        hidden_sizes = [self.hid_dim for i in range(self.num_layer)]
        for ix, nhid in enumerate(hidden_sizes):
            weight = torch.nn.Parameter(torch.FloatTensor(previous_size, nhid).to(self.device))
            bias = torch.nn.Parameter(torch.FloatTensor(nhid).to(self.device))
            w_velocity = torch.zeros(weight.shape).to(self.device)
            b_velocity = torch.zeros(bias.shape).to(self.device)
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)

        output_weight = torch.nn.Parameter(torch.FloatTensor(previous_size, self.out_dim).to(self.device))
        output_w_velocity = torch.zeros(output_weight.shape).to(self.device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        output_bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim).to(self.device))
        output_b_velocity = torch.zeros(output_bias.shape).to(self.device)
        self.biases.append(output_bias)
        self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def self_training(self, features, adj_norm):
        self._initialize()

        for ix in range(self.num_layer + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True

        optimizer = torch.optim.Adam(self.weights + self.biases, lr=self.lr)

        for j in tqdm(range(self.train_iters), desc="self-training"):
            optimizer.zero_grad()
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                hidden = adj_norm @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            optimizer.step()

        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask == 1] = self.y[self.train_mask == 1]

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
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])

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
            if self.with_relu:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden / 1000, dim=1)
        attack_loss = F.nll_loss(output[self.val_mask==1], self.y_self_training[self.val_mask==1])
        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        return adj_grad

    def attack(self):
        # self_training
        features = self.x
        modified_adj = self.adj_ori
        adj_norm = self.normalize_adj_tensor(modified_adj)
        self.inner_train(features, adj_norm)
        self.self_training(features, adj_norm)

        for i in tqdm(range(self.n_perturbations), desc="Perturbing graph"):
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + self.adj_ori

            adj_norm = self.normalize_adj_tensor(modified_adj)
            self.inner_train(features, adj_norm)
            adj_grad = self.get_meta_grad(features, adj_norm)

            smoothed_grad = 0
            smoothed_grad = smoothed_grad + adj_grad

            for loop in range(self.smooth_loop):
                noise = torch.normal(0, self.gauss_noise, size=(features.shape[0], features.shape[1])).to(self.device)
                adj_grad = self.get_meta_grad(features + noise, adj_norm)
                smoothed_grad = smoothed_grad + 0.2 * adj_grad / self.smooth_loop

            adj_meta_grad = smoothed_grad * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad * singleton_mask

            if self.momentum_grad is not None:
                self.momentum_grad = adj_meta_grad.detach() + self.mom * self.momentum_grad
            else:
                self.momentum_grad = adj_meta_grad.detach()

            structural_grad = self.momentum_grad + self.momentum_grad.t()
            structural_grad = structural_grad.reshape([structural_grad.shape[0] * structural_grad.shape[1]])
            grad_abs = torch.abs(structural_grad)

            idxes = torch.argsort(grad_abs, dim=0, descending=True)
            candidate_list = []
            wait_list = []
            grad_sum = []
            _N = smoothed_grad.shape[0]
            for index in range(idxes.shape[0]):
                x = idxes[index] // _N
                x = x.item()
                y = idxes[index] % _N
                y = y.item()
                if (x * _N + y) in candidate_list or (y * _N + x) in candidate_list:
                    continue
                wait_list.append([x, y])
                candidate_list.append(idxes[index].item())

                candidate_adj = modified_adj.clone()
                if len(wait_list) == self.wait_list:
                    interval = self.intervals
                    steps = int(1 / interval)
                    grad_list = [interval * k for k in range(steps)]

                    cand_grad = []
                    for grad_weight in grad_list:
                        for candidate in wait_list:
                            if modified_adj[candidate[0], candidate[1]] == 0:
                                candidate_adj[candidate[0], candidate[1]] = grad_weight
                                candidate_adj[candidate[1], candidate[0]] = grad_weight
                            else:
                                candidate_adj[candidate[0], candidate[1]] = 1 - grad_weight
                                candidate_adj[candidate[1], candidate[0]] = 1 - grad_weight

                        adj_norm = self.normalize_adj_tensor(candidate_adj)
                        adj_grad = self.get_meta_grad(features, adj_norm)
                        adj_grad = adj_grad * (-2 * modified_adj + 1)

                        cand_grad1 = []
                        cand_grad2 = []
                        for candidate in wait_list:
                            cand_grad1.append(adj_grad[candidate[0], candidate[1]].item())
                            cand_grad2.append(adj_grad[candidate[1], candidate[0]].item())
                        del adj_grad
                        gc.collect()
                        cand_grad.append([cand_grad1, cand_grad2])

                    for candidate_idx, candidate in enumerate(wait_list):
                        cand_grad_sum = 0
                        if modified_adj[candidate[0], candidate[1]] == 0:
                            for step in range(steps):
                                cand_grad_sum = cand_grad_sum + interval * (
                                cand_grad[step][0][candidate_idx]) + interval * (cand_grad[step][1][candidate_idx])
                        else:
                            for step in range(steps):
                                cand_grad_sum = cand_grad_sum - interval * (
                                cand_grad[step][0][candidate_idx]) - interval * (cand_grad[step][1][candidate_idx])
                        grad_sum.append(cand_grad_sum)

                    wait_list = []

                    if len(candidate_list) >= self.candidates:
                        order = np.argsort(grad_sum)
                        order = order[::-1]
                        candidate_list = [candidate_list[new_ord] for new_ord in order]
                        break

            row_idx = candidate_list[0] // _N
            col_idx = candidate_list[0] % _N

            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None
        self.modified_features = None