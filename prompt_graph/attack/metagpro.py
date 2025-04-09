import torch
from prompt_graph.attack import PreTrain_task
import torch.nn as nn
from prompt_graph.data import load4node, load4graph, load4node_to_sparse, split_train_val_test
from prompt_graph.model import GAT, GCN, GIN, GraphTransformer, Linearized_GCN
from torch.nn import functional as F
from tqdm import tqdm
from prompt_graph.attack import BaseAttack, PRBCD
import os
import math
from torchmetrics import Accuracy
import copy
import numpy as np
import matplotlib.pyplot as plt
from prompt_graph.utils import edge_adding, edge_dropping, feature_masking, edge_weighted_dropping, identity_augmentation, seed_everything, node_centric_homophily, likelihood_ratio_filter
import random
from torch_geometric.nn.inits import glorot
import warnings

class BasePrompt(BaseAttack):
    def __init__(self, augmentation_list=[identity_augmentation, edge_dropping, edge_adding, feature_masking], *arg,
                 **kwargs):
        super(BasePrompt, self).__init__(*arg, **kwargs)

        if self.attack_features:
            self.feature_changes = torch.nn.Parameter(torch.FloatTensor(self.nnodes, self.input_dim)).to(
                self.device)  # 可以看到如果要改变输入的话 输入也是要求梯度信息的
            self.feature_changes.data.fill_(0)

        if self.attack_structure:
            self.adj_changes = torch.nn.Parameter(torch.FloatTensor(self.nnodes, self.nnodes)).to(self.device)
            self.adj_changes.data.fill_(0)

        self.imitation_model_train(augmentation_list)

    def imitation_model_train(self, augmentation_list):
        """
        the reason why we should train an imitation model is that
        when you use edge_index or torch.SparseTensor as input form of topology in pretrain model created by torch_geometric.nn
        the gradients of adj can not be got.
        Therefore, the imitation model will output the embeddings like pretrain model, but take
        adj as input form of topology.

        If you are interested in bridging the gap between deeoprobust(adj as input) and torch_geometric(edge_index as input),
        see https://github.com/pyg-team/pytorch_geometric/issues/1511
        https://github.com/DSE-MSU/DeepRobust/issues/118

        But we still can not think a by-pass expect this function.
        """
        # we dont want there be so many hyper-parameters,
        # So if you want, you can set surrogate_model, its layer_num and so on as class_number
        self.Linearized_GCN = Linearized_GCN(self.input_dim, self.hid_dim).to(self.device)
        adj_ori_ = copy.deepcopy(self.adj_ori.data.to('cpu'))
        x_ = copy.deepcopy(self.x.data.to('cpu'))

        # here comes an assumption that in the augmentation will probably used in the process of pretraining
        # such as GCL DGI with augmentation
        optimizer = torch.optim.Adam(self.Linearized_GCN.parameters(), lr=0.01, weight_decay=5e-4)
        patience = 10
        for augmentation_func in augmentation_list:
            augmentation = augmentation_func(adj_ori_, x_, 0.2, self.undirected)
            aug_adj, aug_feature = augmentation.augment()
            aug_adj, aug_feature = aug_adj.to(self.device), aug_feature.to(self.device)
            aug_edge_index_0, aug_edge_index_1 = torch.where(aug_adj == 1)
            aug_edge_index = torch.stack([aug_edge_index_0, aug_edge_index_1], dim=0)
            aug_adj_norm = self.normalize_adj_tensor(aug_adj)
            train_loss_min = 1000000
            cnt_wait = 0
            for epoch in tqdm(range(100), desc="training the imitation model with " + augmentation.get_name()):
                optimizer.zero_grad()
                z1 = self.pretrain_gnn(aug_feature, aug_edge_index)
                z2 = self.Linearized_GCN(aug_feature, aug_adj_norm)
                z1 = F.normalize(z1, p=2, dim=1)
                z2 = F.normalize(z2, p=2, dim=1)

                loss = F.mse_loss(z1, z2)
                loss.backward()
                optimizer.step()

                if train_loss_min > loss:
                    train_loss_min = loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait == patience:
                        print('Early stopping at ' + str(epoch) + ' epoch!')
                        break
        self.Linearized_GCN.eval()  # replace pretrain_model with

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

    def save_adj(self, attack_name='MetaGpro'):

        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'adj_mod'}.pt"

        modified_adj = self.get_modified_adj().cpu().clone()

        torch.save(modified_adj, os.path.join(self.save_fold_path, name))


    def save_feature(self, attack_name='MetaGpro'):

        assert self.modified_features is not None, \
                'modified_adj is None! Please perturb the feature first.'
        
        name = f"{self.target_dataset}/{attack_name}.{self.budget+'budget'}.{'feature_mod'}.pt"

        modified_features = self.get_modified_features().cpu().clone()

        torch.save(modified_features, os.path.join(self.save_fold_path, name))


class MetaGpro_Approx(BasePrompt):
    def __init__(self, train_iters: int = 100, lr: float = 0.1, momentum: float = 0.9, lambda_: float = 0.5, budget: float = 0.05, 
                save_fold_path="/root/autodl-tmp/deeprobust", surrogate_token_num=10, surrogate_prompt='Two-views',attack_loss='CE', 
                aug_ratio=0.2, all_in_one_threshold=0.5, lenda_1=0.1, lenda_2=0.4, vic_coef1=1.0, vic_coef2=1.0, vic_coef3=0.04, *arg, **kwargs):
        super(MetaGpro_Approx, self).__init__(*arg, **kwargs)

        self.momentum = momentum
        self.lr = lr

        self.train_iters = train_iters
        self.save_fold_path = save_fold_path
        self.token_num = surrogate_token_num

        self.train_acc_list = []
        self.val_acc_list = []
        self.train_loss_list = []

        if self.attack_structure:
            self.adj_changes.retain_grad()
            self.adj_grad_sum = torch.zeros(self.nnodes, self.nnodes).to(self.device)
        if self.attack_features:
            self.feature_changes.retain_grad()
            self.feature_grad_sum = torch.zeros(self.nnodes, self.input_dim).to(self.device)

        self.all_in_one_threshold = all_in_one_threshold

        assert 0 <= lambda_ <= 1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_  # lambda * label_loss + (1-lambda) * ublabeled
        self.lenda_1 = lenda_1  # loss weight for 'Infomax for agreement maximization'
        self.lenda_2 = lenda_2  # loss weight for 'Infomax for over-robust enhancement'
        self.vic_coef1 = vic_coef1  # weight for one part in constrative learning loss
        self.vic_coef2 = vic_coef2  # weight for one part in constrative learning loss
        self.vic_coef3 = vic_coef3  # weight for one part in constrative learning loss

        self.aug_ratio = aug_ratio
        self.budget = budget

        if self.undirected:
            self.n_perturbations = int(self.nedges * budget // 2)
        else:
            self.n_perturbations = int(self.nedges * budget)

        self.weights = []

        self.attack_loss_name = attack_loss
        self.attack_loss_functions(attack_loss)

        self.surrogate_prompt = surrogate_prompt
        self.surrogate_prompt_tokens(surrogate_prompt)

        answer_linear = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.out_dim).to(self.device))  # answer head
        answer_bias = torch.nn.Parameter(torch.Tensor(1, self.out_dim).to(self.device))
        self.weights.append(answer_linear)
        self.weights.append(answer_bias)

        self._initialize()

    def _initialize(self):

        for w in self.weights:
            glorot(w)

        self.optimizer = torch.optim.Adam(self.weights, lr=self.lr)

    def self_training(self, features, adj_norm):

        self._initialize()

        train_loss_min = 100000
        cnt_wait = 0
        for j in tqdm(range(400), desc="self-training"):
            self.optimizer.zero_grad()
            hidden, _ = self.surrogate_prompt_forward_func(features, adj_norm)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            self.optimizer.step()
            
            if train_loss_min > loss_labeled:
                train_loss_min = loss_labeled
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == 20:
                    print('Early stopping at ' + str(j) + ' epoch!')
                    break
        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.val_mask == 1], self.y[self.val_mask == 1]).item()
        print("you can use self-training to judge whether the hyper-parameter settings, such as lr, are reasonable or not")
        print("self-training acc on labeld_data:", labeled_data_acc)
        print("self-training acc on val_data:", unlabeld_data_acc)

        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask == 1] = self.y[self.train_mask == 1]

        #print(self.train_loss_list)
        """print(self.train_loss_list)
        train_loss = np.array(self.train_loss_list)
        folder_path = "./result"
        file_path = os.path.join(folder_path, "train_loss_view1_20_num_mean_0.3.npy")
        np.save(file_path, train_loss)"""

    def inner_train(self, features, modified_adj):
        "we find momentum in Mettack can't learn prompt well, so we use trick in MetaApprox"

        adj_norm = self.normalize_adj_tensor(modified_adj)
        viewlearner = PRBCD(features, modified_adj, self.Linearized_GCN, self.surrogate_prompt, self.train_mask, 10000,
                            self.undirected, self.device, self.token_num, self.vic_coef1, self.vic_coef2,
                            self.vic_coef3,
                            self.aug_ratio, 0.1)
        for j in range(self.train_iters):
            self.optimizer.zero_grad()
            hidden, loss_agreement_infomax = self.surrogate_prompt_forward_func(features, adj_norm)

            if self.lenda_2 > 0.005:
                weight_datas = []
                for weight in self.weights:
                    weight_datas.append(weight.data)
                if j % 5 ==0:  # 每个循环都forward效率有点低 Forwarding every loop is inefficient
                    viewlearner.viewer_train_forward(weight_datas, hidden.data)
                loss_cons = viewlearner.viewer_eval_forward(self.weights, hidden)
            else:
                loss_cons = 0.0

            output = F.log_softmax(hidden, dim=1)

            loss_CE = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_surrogate = loss_CE + self.lenda_1 * loss_agreement_infomax + self.lenda_2 * loss_cons

            loss_labeled = self.attack_loss_func(output, self.y, self.train_mask==1)
            loss_unlabeled = self.attack_loss_func(output, self.y_self_training, self.train_mask==0)
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
            #print("check:", loss_surrogate)
            loss_surrogate.backward(retain_graph=True)

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

        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else self.adj_ori
        self.modified_features = self.get_modified_features().detach() if self.attack_features else self.x

    def augmented_view(self, features, adj_norm):  # 注意这里features和adj都是modified后的, 且adj不是norm后的
        pass

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
        loss2 = k * torch.tanh(1/k * -margin[margin < 0])
        loss = torch.cat([loss1, loss2])

        return loss.mean()

    def MCE_loss(self, logits, labels, index):
        logits_, labels_ = logits[index], labels[index]
        not_flipped = logits_.argmax(-1) == labels_
        return -F.nll_loss(logits_[not_flipped], labels_[not_flipped])

    def surrogate_prompt_tokens(self, surrogate_prompt):
        prompt_token1 = None
        prompt_token2 = None
        gnn_weight1 = None
        gnn_weight2 = None
        if surrogate_prompt == 'Two-views':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))  # "GPF"  tokens
            prompt_token2 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))    # "GraphPrompt" tokens

        elif surrogate_prompt in ['All-in-one', 'All-in-one-mean', 'All-in-one-softmax']:
            prompt_token1 = torch.nn.Parameter(torch.Tensor(self.token_num, self.input_dim).to(self.device))

        elif surrogate_prompt == 'Gprompt':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))


        elif surrogate_prompt == 'GPF':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))

        elif surrogate_prompt == 'GPF-plus':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.token_num).to(self.device))
            prompt_token2 = torch.nn.Parameter(torch.Tensor(self.token_num, self.input_dim).to(self.device))

        elif surrogate_prompt == 'GPF-GNN':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))
            gnn_weight1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hid_dim).to(self.device))
            gnn_weight2 = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim).to(self.device))

        elif surrogate_prompt == 'Gprompt-GNN':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))
            gnn_weight1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hid_dim).to(self.device))
            gnn_weight2 = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim).to(self.device))


        self.weights.append(prompt_token1)

        for weight in [prompt_token2, gnn_weight1, gnn_weight2]:
            if weight is not None:
                self.weights.append(weight)

        forward_func_dict = {'Two-views': self.two_views_forward, 'All-in-one': self.all_in_one_forward,'All-in-one-softmax':self.smooth_all_in_one_forward,
                             'All-in-one-mean': self.sparse_all_in_one_forward, 'GPF': self.gpf_forward,
                             'GPF-plus': self.gpf_plus_forward, 'Gprompt': self.gprompt_forward,
                             'GPF-GNN': self.gpf_gnn_forward, 'Gprompt-GNN': self.gprompt_gnn_forward}

        self.surrogate_prompt_forward_func = forward_func_dict[surrogate_prompt]
    
    def gpf_gnn_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        support1 = torch.mm(prompted_features, self.weights[1])
        output1 = torch.mm(adj_norm, support1)      
        support2 = torch.mm(output1, self.weights[2])
        output2= torch.mm(adj_norm, support2)

        hidden = output2 @ self.weights[3] + self.weights[4]     
        return hidden, 0

    def gprompt_gnn_forward(self, features, adj_norm):
        support1 = torch.mm(features, self.weights[1])
        output1 = torch.mm(adj_norm, support1)      
        support2 = torch.mm(output1, self.weights[2])
        output2= torch.mm(adj_norm, support2)
        prompted_hidden = output2 * self.weights[0]

        hidden = prompted_hidden @ self.weights[3] + self.weights[4]     
        return hidden, 0


    def two_views_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        hidden2 = self.Linearized_GCN(features, adj_norm)

        prompted_hidden1 = self.Linearized_GCN(prompted_features, adj_norm)
        prompted_hidden2 = hidden2 * self.weights[1]

        hidden = 1 / 2 * (prompted_hidden1 + prompted_hidden2) @ self.weights[2] + self.weights[3]

        loss_agreement_infomax = F.mse_loss(prompted_hidden1, prompted_hidden2)  # only consider positive example, its no need to push one away from another
        return hidden, loss_agreement_infomax

    def smooth_all_in_one_forward(self, features, adj_norm):
        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))     # (n_nodes, token_nums)
        weight = torch.softmax(weight, dim=1)
        weighted_prompt_tokens = torch.mm(weight, self.weights[0])    # (n_nodes, input_dim)

        prompted_features = features  + weighted_prompt_tokens
        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def sparse_all_in_one_forward(self, features, adj_norm):
        #assert 0.4 <= threshold <= 1, "In Sparse-All-in-one, default 0.4<= threshold <=1"
        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.all_in_one_threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.weights[0])  # (n_nodes, input_dim)

        # the prompted function in "All in one" is x' = x + sum(pk)
        # and we found in surrogate module, change it to x' = x + mean(pk) is more stable in learning
        # we guess the reason behind this phenomenon is if token_num is large, then x' is far away from x at beginning
        # prompted_features = features + weighted_prompt_tokens
        prompted_features = features + 1 / self.token_num * weighted_prompt_tokens

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def all_in_one_forward(self, features, adj_norm):
        # use 'sparse all_in_one',which has adventages of speed, flexibility, Stable gradient descent process
        #assert 0 <= threshold <= 1, "please set 0<=threshold<=1"

        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.all_in_one_threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.weights[0])  # (n_nodes, input_dim)

        prompted_features = features + weighted_prompt_tokens

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gprompt_forward(self, features, adj_norm):
        embeddings = self.Linearized_GCN(features, adj_norm)
        prompted_hidden = embeddings * self.weights[0]

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gpf_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gpf_plus_forward(self, features, adj_norm):
        score = features @ self.weights[0]
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.weights[1])

        prompted_features = features + p

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[2] + self.weights[3]
        return hidden, 0

    def plot_acc_during_training(self):
        print("Warning: In general, graph Prompt learning converges more slowly than learning of GNNs,so observing acc after fewer iterations is less accurate")
        if not os.path.exists("./result"):
            os.makedirs("./result")
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

        save_file_name = f"MetaGpro_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)
    
    def save_train_loss(self):
        train_loss = np.array(self.train_loss_list)
        folder_path = "./result"+ "/" + self.surrogate_prompt
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.surrogate_prompt in ['All-in-one', 'All-in-one-mean', 'All-in-one-softmax']:
            file_name = f"train_loss_{self.target_dataset}_{self.surrogate_prompt}_{str(self.token_num)+'token_num'}+{str(self.all_in_one_threshold)+'threshold'}.npy"
        else:
            file_name = f"train_loss_{self.target_dataset}_{self.surrogate_prompt}_{str(self.token_num)+'token_num'}.npy"
        file_path = os.path.join(folder_path, file_name)
        np.save(file_path, train_loss)


    def Retrain_and_test(self):
        """
        After attack, retrain the prompt and check acc on test dataset.
        """
        self._initialize()

        train_loss_min = 100000
        cnt_wait = 0

        features = self.modified_features
        adj_norm = self.normalize_adj_tensor(self.modified_adj)

        for j in tqdm(range(400), desc="self-training"):
            self.optimizer.zero_grad()
            hidden, _ = self.surrogate_prompt_forward_func(features, adj_norm)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            self.optimizer.step()

            #self.train_loss_list.append(loss_labeled.item())
            if train_loss_min > loss_labeled:
                train_loss_min = loss_labeled
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == 10:
                    print('Early stopping at ' + str(j) + ' epoch!')
                    break
        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.test_mask == 1], self.y[self.test_mask == 1]).item()
        print("After attack, retrain and test")
        print("self-training acc on labeld_data:", labeled_data_acc)
        print("self-training acc on test_data:", unlabeld_data_acc)


class MetaGpro(BasePrompt):
    def __init__(self, train_iters: int = 100, lr: float = 0.01, lambda_: float = 0.5,budget: float = 0.05, 
                save_fold_path="/root/autodl-tmp/deeprobust", surrogate_token_num=10, surrogate_prompt='Two-views',attack_loss='CE', 
                aug_ratio=0.2, all_in_one_threshold=0.5,  lenda_1=0.0, lenda_2=0.4, vic_coef1=1.0, vic_coef2=1.0,
                vic_coef3=0.04, *arg, **kwargs):
        super(MetaGpro, self).__init__(*arg, **kwargs)

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = lr

        self.train_iters = train_iters
        self.save_fold_path = save_fold_path
        self.token_num = surrogate_token_num

        self.train_acc_list = []
        self.val_acc_list = []
        self.train_loss_list = []

        if self.attack_structure:
            self.adj_changes.retain_grad()
            self.adj_grad_sum = torch.zeros(self.nnodes, self.nnodes).to(self.device)
        if self.attack_features:
            self.feature_changes.retain_grad()
            self.feature_grad_sum = torch.zeros(self.nnodes, self.input_dim).to(self.device)

        self.all_in_one_threshold = all_in_one_threshold

        assert 0 <= lambda_ <= 1, "lambda_ should between 0 and 1"
        self.lambda_ = lambda_  # lambda * label_loss + (1-lambda) * ublabeled
        self.lenda_1 = lenda_1  # loss weight for 'Infomax for agreement maximization'
        self.lenda_2 = lenda_2  # loss weight for 'Infomax for over-robust enhancement'
        self.vic_coef1 = vic_coef1  # weight for one part in constrative learning loss
        self.vic_coef2 = vic_coef2  # weight for one part in constrative learning loss
        self.vic_coef3 = vic_coef3  # weight for one part in constrative learning loss

        self.aug_ratio = aug_ratio
        self.budget = budget

        if self.undirected:
            self.n_perturbations = int(self.nedges * budget // 2)
        else:
            self.n_perturbations = int(self.nedges * budget)

        self.weights = []

        self.attack_loss_name = attack_loss
        self.attack_loss_functions(attack_loss)

        self.surrogate_prompt = surrogate_prompt
        self.surrogate_prompt_tokens(surrogate_prompt)

        answer_linear = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.out_dim).to(self.device))  # answer head
        answer_bias = torch.nn.Parameter(torch.Tensor(1, self.out_dim).to(self.device))
        self.weights.append(answer_linear)
        self.weights.append(answer_bias)

        self.weights_cut = []
        self.weights_m = []
        self.weights_v = []
        for w in self.weights:
            self.weights_m.append(torch.zeros(w.shape).to(self.device))
            self.weights_v.append(torch.zeros(w.shape).to(self.device))

        self._initialize()

    def _initialize(self):
        
        for w, m, v in zip(self.weights, self.weights_m, self.weights_v):
            glorot(w)
            m.data.fill_(0)
            v.data.fill_(0)

    def cut_initialize(self):   # If the beginning of inner training is not stable, use this func
        for w, c, m, v in zip(self.weights,self.weights_cut, self.weights_m, self.weights_v):
            w = c
            m.data.fill_(0)
            v.data.fill_(0)

    def self_training(self, features, modified_adj):
        self._initialize()
        adj_norm = self.normalize_adj_tensor(modified_adj)
        train_loss_min = 100000
        cnt_wait = 0
        optimizer = torch.optim.Adam(self.weights, lr=self.lr)
        for j in tqdm(range(400), desc="self-training"):
            optimizer.zero_grad()
            hidden, _ = self.surrogate_prompt_forward_func(features, adj_norm)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            optimizer.step()
    
            if j == 99:
                print(loss_labeled)
                for ix in range(len(self.weights)):
                    self.weights_cut.append(self.weights[ix].data)  

            #self.train_loss_list.append(loss_labeled.item())
            if train_loss_min > loss_labeled:
                train_loss_min = loss_labeled
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == 20:
                    print('Early stopping at ' + str(j) + ' epoch!')
                    break
        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.val_mask == 1], self.y[self.val_mask == 1]).item()
        print("you can use self-training to judge whether the hyper-parameter settings, such as lr, are reasonable or not")
        print("self-training acc on labeld_data:", labeled_data_acc)
        print("self-training acc on val_data:", unlabeld_data_acc)

        self.y_self_training = output.argmax(1)
        self.y_self_training[self.train_mask == 1] = self.y[self.train_mask == 1]

        #print(self.train_loss_list)
        """print(self.train_loss_list)
        train_loss = np.array(self.train_loss_list)
        folder_path = "./result"
        file_path = os.path.join(folder_path, "train_loss_view1_20_num_mean_0.3.npy")
        np.save(file_path, train_loss)"""

    def inner_train(self, features, modified_adj):
        "we find momentum in Mettack can't learn prompt well, so we use trick in MetaApprox"
        #self._initialize()
        self.cut_initialize()
        adj_norm = self.normalize_adj_tensor(modified_adj)
        iter_ = 0
        viewlearner = PRBCD(features, modified_adj, self.Linearized_GCN, self.surrogate_prompt, self.train_mask, 10000,
                            self.undirected, self.device, self.token_num, self.vic_coef1, self.vic_coef2,
                            self.vic_coef3,
                            self.aug_ratio, 0.1)
                            
        for ix in range(len(self.weights)):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.weights_m[ix] = self.weights_m[ix].detach()
            self.weights_m[ix].requires_grad = True
            self.weights_v[ix] = self.weights_v[ix].detach()
            self.weights_v[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden, loss_agreement_infomax = self.surrogate_prompt_forward_func(features, adj_norm)

            if self.lenda_2 > 0.005:
                weight_datas = []
                for weight in self.weights:
                    weight_datas.append(weight.data)
                if j % 5 ==0:  # 每个循环都forward效率有点低 Forwarding every loop is inefficient
                    viewlearner.viewer_train_forward(weight_datas, hidden.data)
                loss_cons = viewlearner.viewer_eval_forward(self.weights, hidden)
            else:
                loss_cons = 0.0

            output = F.log_softmax(hidden, dim=1)

            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_unlabeled = F.nll_loss(output[self.train_mask == 0], self.y_self_training[self.train_mask == 0])
            loss_surrogate = loss_labeled + self.lenda_1 * loss_agreement_infomax + self.lenda_2 * loss_cons
            
            iter_ += 1
            lr_t = self.lr * math.sqrt(1.0 - self.beta2**iter_) / (1.0 - self.beta1**iter_)
            
            weight_grads = torch.autograd.grad(loss_surrogate, self.weights)
            self.weights_m = [ m + (1 - self.beta1) * (g - m) for m, g in zip(self.weights_m, weight_grads)]
            self.weights_v = [ v + (1 - self.beta2) * (g**2 - v) for v, g in zip(self.weights_v, weight_grads)]
            self.weights = [w - lr_t * m / (torch.sqrt(v) + 1e-8) for w, m, v in zip(self.weights, self.weights_m, self.weights_v)]

    

    def get_meta_grad(self, features, modified_adj):
        adj_norm = self.normalize_adj_tensor(modified_adj)
        
        hidden, _ = self.surrogate_prompt_forward_func(features, adj_norm)
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
        modified_features = self.x
        modified_adj = self.adj_ori
        self.self_training(modified_features, modified_adj)

        for i in tqdm(range(self.n_perturbations), desc="Perturbing graph"):
            #self._initialize()
            if self.attack_structure:
                modified_adj = self.get_modified_adj()
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = self.get_modified_features()
                self.feature_grad_sum.data.fill_(0)

            self.inner_train(modified_features, modified_adj)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, modified_adj)
            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

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

        self.modified_adj = self.get_modified_adj().detach() if self.attack_structure else None
        self.modified_features = self.get_modified_features().detach() if self.attack_features else None

    def augmented_view(self, features, adj_norm):  # 注意这里features和adj都是modified后的, 且adj不是norm后的
        pass

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
        loss2 = k * torch.tanh(k * -margin[margin < 0])      # we find this better than one in the papar
        loss = torch.cat([loss1, loss2])

        return loss.mean()

    def MCE_loss(self, logits, labels, index):
        logits_, labels_ = logits[index], labels[index]
        not_flipped = logits_.argmax(-1) == labels_
        return -F.nll_loss(logits_[not_flipped], labels_[not_flipped])

    def surrogate_prompt_tokens(self, surrogate_prompt):
        prompt_token1 = None
        prompt_token2 = None
        gnn_weight1 = None
        gnn_weight2 = None
        if surrogate_prompt == 'Two-views':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))  # "GPF"  tokens
            prompt_token2 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))    # "GraphPrompt" tokens

        elif surrogate_prompt in ['All-in-one', 'All-in-one-mean', 'All-in-one-softmax']:
            prompt_token1 = torch.nn.Parameter(torch.Tensor(self.token_num, self.input_dim).to(self.device))

        elif surrogate_prompt == 'Gprompt':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))


        elif surrogate_prompt == 'GPF':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))

        elif surrogate_prompt == 'GPF-plus':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.token_num).to(self.device))
            prompt_token2 = torch.nn.Parameter(torch.Tensor(self.token_num, self.input_dim).to(self.device))

        elif surrogate_prompt == 'GPF-GNN':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.input_dim).to(self.device))
            gnn_weight1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hid_dim).to(self.device))
            gnn_weight2 = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim).to(self.device))

        elif surrogate_prompt == 'Gprompt-GNN':
            prompt_token1 = torch.nn.Parameter(torch.Tensor(1, self.hid_dim).to(self.device))
            gnn_weight1 = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hid_dim).to(self.device))
            gnn_weight2 = torch.nn.Parameter(torch.Tensor(self.hid_dim, self.hid_dim).to(self.device))


        self.weights.append(prompt_token1)

        for weight in [prompt_token2, gnn_weight1, gnn_weight2]:
            if weight is not None:
                self.weights.append(weight)

        forward_func_dict = {'Two-views': self.two_views_forward, 'All-in-one': self.all_in_one_forward,'All-in-one-softmax':self.smooth_all_in_one_forward,
                             'All-in-one-mean': self.sparse_all_in_one_forward, 'GPF': self.gpf_forward,
                             'GPF-plus': self.gpf_plus_forward, 'Gprompt': self.gprompt_forward,
                             'GPF-GNN': self.gpf_gnn_forward, 'Gprompt-GNN': self.gprompt_gnn_forward}

        self.surrogate_prompt_forward_func = forward_func_dict[surrogate_prompt]
    
    def gpf_gnn_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        support1 = torch.mm(prompted_features, self.weights[1])
        output1 = torch.mm(adj_norm, support1)      
        support2 = torch.mm(output1, self.weights[2])
        output2= torch.mm(adj_norm, support2)

        hidden = output2 @ self.weights[3] + self.weights[4]     
        return hidden, 0

    def gprompt_gnn_forward(self, features, adj_norm):
        support1 = torch.mm(features, self.weights[1])
        output1 = torch.mm(adj_norm, support1)      
        support2 = torch.mm(output1, self.weights[2])
        output2= torch.mm(adj_norm, support2)
        prompted_hidden = output2 * self.weights[0]

        hidden = prompted_hidden @ self.weights[3] + self.weights[4]     
        return hidden, 0


    def two_views_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        hidden2 = self.Linearized_GCN(features, adj_norm)

        prompted_hidden1 = self.Linearized_GCN(prompted_features, adj_norm)
        prompted_hidden2 = hidden2 * self.weights[1]

        hidden = 1 / 2 * (prompted_hidden1 + prompted_hidden2) @ self.weights[2] + self.weights[3]

        loss_agreement_infomax = F.mse_loss(prompted_hidden1, prompted_hidden2)  # only consider positive example, its no need to push one away from another
        return hidden, loss_agreement_infomax

    def smooth_all_in_one_forward(self, features, adj_norm):
        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))     # (n_nodes, token_nums)
        weight = torch.softmax(weight, dim=1)
        weighted_prompt_tokens = torch.mm(weight, self.weights[0])    # (n_nodes, input_dim)

        prompted_features = features  + weighted_prompt_tokens
        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0
        
    def sparse_all_in_one_forward(self, features, adj_norm):
        #assert 0.4 <= threshold <= 1, "In All-in-one-mean, default 0.4<= threshold <=1"

        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.all_in_one_threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.weights[0])  # (n_nodes, input_dim)

        # the prompted function in "All in one" is x' = x + sum(pk)
        # and we found in surrogate module, change it to x' = x + mean(pk) is more stable in learning
        # we guess the reason behind this phenomenon is if token_num is large, then x' is far away from x at beginning
        # prompted_features = features + weighted_prompt_tokens
        prompted_features = features + 1 / self.token_num * weighted_prompt_tokens

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def all_in_one_forward(self, features, adj_norm):
        # use 'sparse all_in_one',which has adventages of speed, flexibility, Stable gradient descent process
        #assert 0 <= threshold <= 1, "please set 0<=threshold<=1"

        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < self.all_in_one_threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, self.weights[0])  # (n_nodes, input_dim)

        prompted_features = features + weighted_prompt_tokens

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gprompt_forward(self, features, adj_norm):
        embeddings = self.Linearized_GCN(features, adj_norm)
        prompted_hidden = embeddings * self.weights[0]

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gpf_forward(self, features, adj_norm):
        prompted_features = features + self.weights[0]
        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden, 0

    def gpf_plus_forward(self, features, adj_norm):
        score = features @ self.weights[0]
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.weights[1])

        prompted_features = features + p

        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[2] + self.weights[3]
        return hidden, 0

    def plot_acc_during_training(self):
        print("Warning: In general, graph Prompt learning converges more slowly than learning of GNNs,so observing acc after fewer iterations is less accurate")
        if not os.path.exists("./result"):
            os.makedirs("./result")
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

        save_file_name = f"MetaGpro_for_{self.target_dataset}_with_{self.n_perturbations}.png"
        path = os.path.join("./result", save_file_name)
        plt.savefig(path)

    def save_train_loss(self):
        train_loss = np.array(self.train_loss_list)
        folder_path = "./result"+ "/" + self.surrogate_prompt
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.surrogate_prompt in ['All-in-one', 'All-in-one-mean', 'All-in-one-softmax']:
            file_name = f"train_loss_{self.target_dataset}_{self.surrogate_prompt}_{str(self.token_num)+'token_num'}+{str(self.all_in_one_threshold)+'threshold'}.npy"
        else:
            file_name = f"train_loss_{self.target_dataset}_{self.surrogate_prompt}_{str(self.token_num)+'token_num'}.npy"
        file_path = os.path.join(folder_path, file_name)
        np.save(file_path, train_loss)


    def Retrain_and_test(self):
        """
        After attack, retrain the prompt and check acc on test dataset.
        """
        self._initialize()

        train_loss_min = 100000
        cnt_wait = 0

        features = self.modified_features
        adj_norm = self.normalize_adj_tensor(self.modified_adj)

        for j in tqdm(range(400), desc="self-training"):
            self.optimizer.zero_grad()
            hidden, _ = self.surrogate_prompt_forward_func(features, adj_norm)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.train_mask == 1], self.y[self.train_mask == 1])
            loss_labeled.backward()
            self.optimizer.step()

            """if j<100:
                self.train_loss_list.append(loss_labeled.item())"""

            if train_loss_min > loss_labeled:
                train_loss_min = loss_labeled
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == 10:
                    print('Early stopping at ' + str(j) + ' epoch!')
                    break
        accuracy = Accuracy(num_classes=self.out_dim, task='multiclass').to(self.device)
        labeled_data_acc = accuracy(output[self.train_mask == 1], self.y[self.train_mask == 1]).item()
        unlabeld_data_acc = accuracy(output[self.test_mask == 1], self.y[self.test_mask == 1]).item()
        print("After attack, retrain and test")
        print("self-training acc on labeld_data:", labeled_data_acc)
        print("self-training acc on test_data:", unlabeld_data_acc)



def Test():
    # surrogate_prompt_list = ['Two-views', 'All-in-one', 'All-in-one-mean', 'All-in-one-softmax', 'GPF', 'GPF-plus', 'Gprompt', 'GPF-GNN', 'Gprompt-GNN']
    # attack_loss_list = ['CE','MCE','Tanh','Bias_Tanh', 'GraD']
    for target_dataset in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if target_dataset in ['Cora', 'CiteSeer', 'PubMed']:
            seed_everything(42)
            model = MetaGpro(pretrain_dataset='PubMed',target_dataset=target_dataset, surrogate_prompt='GPF', 
                            token_num=10, lr=0.1, train_iters=100, lenda_2=0.0, budget=0.001)
            model.attack()
            #model.visualize_topology_attack_influence()
        else:
            seed_everything(42)
            model = MetaGpro(pretrain_dataset='Computers',target_dataset=target_dataset, surrogate_prompt='GPF', all_in_one_threshold=0.7,
                            token_num=10, lr=0.1, train_iters=100, lenda_2=0.0, budget=0.001)
            
            model.attack()
            #model.visualize_topology_attack_influence()
    #model.visualize_topology_attack_tendency()
    #model.plot_acc_during_training()


if __name__ == '__main__':
    Test()