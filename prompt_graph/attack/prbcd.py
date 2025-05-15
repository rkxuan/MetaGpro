import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch_sparse
from torch_sparse import coalesce
import math


class PRBCD:
    # used for over-robust constrative learning
    # And if you are intersted in Deeprobust in Graph, paper "Robustness of Graph Neural Networks at Scale" is a solid work.
    def __init__(self, features, adj, gnn_model, surrogate_forward_type, train_mask, search_space_size=100000,
                 undirected=True, device='cuda', token_num=None, vic_coef1=1.0, vic_coef2=1.0, vic_coef3=0.04,
                 aug_ratio=0.1, lr_adj=0.1, do_synchronize=True):

        self.features = features
        self.adj = adj

        self.gnn_model = gnn_model
        self.gnn_model.eval()

        forward_func_dict = {'Two-views': self.two_views_forward, 'All-in-one': self.all_in_one_forward,
                             'All-in-one-softmax': self.smooth_all_in_one_forward,
                             'All-in-one-mean': self.sparse_all_in_one_forward, 'GPF': self.gpf_forward,
                             'GPF-plus': self.gpf_plus_forward, 'Gprompt': self.gprompt_forward,
                             'GPF-GNN': self.gpf_gnn_forward, 'Gprompt-GNN': self.gprompt_gnn_forward}

        self.surrogate_forward_func = forward_func_dict[surrogate_forward_type]
        self.train_mask = train_mask
        self.search_space_size = search_space_size
        self.undirected = undirected

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.modified_edge_index = None
        self.perturbed_edge_weight = None

        edge_index_0, edge_index_1 = torch.where(self.adj == 1)
        self.edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.nnodes = adj.shape[0]
        nedges = self.edge_index.shape[1] - self.nnodes  # delete self-loop

        self.n_perturbations = int(aug_ratio * nedges // 2) if self.undirected else int(aug_ratio * nedges)

        self.token_num = token_num

        self.vic_coef1 = vic_coef1
        self.vic_coef2 = vic_coef2
        self.vic_coef3 = vic_coef3

        self.eps = 1e-7
        self.lr_adj = lr_adj

        if self.undirected:
            self.n_possible_edges = self.nnodes * (self.nnodes - 1) // 2
        else:
            self.n_possible_edges = self.nnodes ** 2  # We filter self-loops later

        # self.with_early_stopping = with_early_stopping
        self.do_synchronize = do_synchronize

        self.sample_random_block()

    def viewer_train_forward(self, weights, z1):
        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=self.lr_adj)

        modified_edge_index, modified_edge_weight = self.get_modified_adj()  # serve as augmentation view
        modified_adj = torch.zeros([self.nnodes, self.nnodes]).to(self.device)
        modified_adj[modified_edge_index[0, :], modified_edge_index[1, :]] = modified_edge_weight
        modified_adj_norm = self.normalize_adj_tensor(modified_adj)
        # modified_adj_norm[modified_edge_index[0, :], modified_edge_index[1, :]] *= modified_edge_weight

        z2 = self.surrogate_forward_func(self.features, modified_adj_norm, weights)

        z1 = z1[self.train_mask == 0]
        z2 = z2[self.train_mask == 0]

        # metacon_d has much less computational cost than metacon_s, so we use this trick
        bs, num_feat = z1.shape
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
        cov_loss = self.off_diagonal(cov_z1).pow_(2).sum().div(num_feat) + self.off_diagonal(cov_z2).pow_(2).sum().div(
            num_feat)

        loss_unlabeled = self.vic_coef1 * mse_loss + self.vic_coef2 * std_loss + self.vic_coef3 * cov_loss

        gradient = self.grad_with_checkpoint(loss_unlabeled, self.perturbed_edge_weight)[0]

        if torch.cuda.is_available() and self.do_synchronize:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with torch.no_grad():
            self.updata_edge_weights(gradient)  # will call optimizer.zero_grad()
            self.perturbed_edge_weight = self.project(self.perturbed_edge_weight)

            del modified_edge_index, modified_edge_weight

            self.resample_random_block()

            # edge_index, edge_weight = self.get_modified_adj()

    def viewer_eval_forward(self, weights, z1):
        self.perturbed_edge_weight.requires_grad = False

        modified_edge_index, modified_edge_weight = self.get_modified_adj()  # serve as augmentation view
        modified_adj = torch.zeros([self.nnodes, self.nnodes]).to(self.device)
        modified_adj[modified_edge_index[0, :], modified_edge_index[1, :]] = modified_edge_weight
        modified_adj_norm = self.normalize_adj_tensor(modified_adj)
        # modified_adj_norm[modified_edge_index[0, :], modified_edge_index[1, :]] *= modified_edge_weight

        z2 = self.surrogate_forward_func(self.features, modified_adj_norm, weights)

        z1 = z1[self.train_mask == 0]
        z2 = z2[self.train_mask == 0]

        # metacon_d has much less computational cost than metacon_s, so we use this trick
        bs, num_feat = z1.shape
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
        cov_loss = self.off_diagonal(cov_z1).pow_(2).sum().div(num_feat) + self.off_diagonal(cov_z2).pow_(
            2).sum().div(num_feat)

        loss_unlabeled = self.vic_coef1 * mse_loss + self.vic_coef2 * std_loss + self.vic_coef3 * cov_loss

        return loss_unlabeled

    def updata_edge_weights(self, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = -gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def project(self, values, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > self.n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, self.n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=self.eps, max=1 - self.eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=self.eps, max=1 - self.eps
            ))
        return values

    def get_modified_adj(self):
        if self.undirected:
            modified_edge_index, modified_edge_weight = self.to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight
        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.nnodes, n=self.nnodes, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    def to_symmetric(self, edge_index, edge_weight, op='mean'):
        symmetric_edge_index = torch.cat(
            (edge_index, edge_index.flip(0)), dim=-1
        )

        symmetric_edge_weight = edge_weight.repeat(2)

        symmetric_edge_index, symmetric_edge_weight = coalesce(
            symmetric_edge_index,
            symmetric_edge_weight,
            m=self.nnodes,
            n=self.nnodes,
            op=op
        )

        return symmetric_edge_index, symmetric_edge_weight

    def sample_random_block(self):
        for _ in range(10):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.undirected:
                self.modified_edge_index = self.linear_to_triu_idx(self.nnodes, self.current_search_space)
            else:
                self.modified_edge_index = self.linear_to_full_idx(self.nnodes, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
        )
        if self.current_search_space.shape[0] >= self.n_perturbations:
            return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def resample_random_block(self):
        self.keep_heuristic = 'WeightOnly'
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.shape[0] // 2:
                idx_keep = sorted_idx.shape[0] // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(10):
            n_edges_resample = self.search_space_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.undirected:
                self.modified_edge_index = self.linear_to_triu_idx(self.nnodes, self.current_search_space)
            else:
                self.modified_edge_index = self.linear_to_full_idx(self.nnodes, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
            ] = perturbed_edge_weight_old  # unique_idx: the indices for the old edges

            if self.undirected is False:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > self.n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')

    def linear_to_full_idx(self, n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    def linear_to_triu_idx(self, n: int, lin_idx: torch.Tensor) -> torch.Tensor:
        row_idx = (
                n
                - 2
                - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
                lin_idx
                + row_idx
                + 1 - n * (n - 1) // 2
                + (n - row_idx) * ((n - row_idx) - 1) // 2
        )
        return torch.stack((row_idx, col_idx))

    def normalize_adj_tensor(self, adj):
        # adj_ = adj + torch.eye(adj.shape[0])      不用添加对角线元素， 因为加载数据的时候预处理过了
        D = torch.sum(adj, dim=1)
        D_inv = torch.pow(D, -1 / 2)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat_inv = torch.diag(D_inv)

        adj_norm = D_mat_inv @ adj @ D_mat_inv  # GCN的归一化方式
        return adj_norm

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def grad_with_checkpoint(self, outputs, inputs):
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
        for input in inputs:
            if not input.is_leaf:
                input.retain_grad()
        torch.autograd.backward(outputs)

        grad_outputs = []
        for input in inputs:
            grad_outputs.append(input.grad.clone())
            input.grad.zero_()
        return grad_outputs

    def gpf_gnn_forward(self, features, adj_norm, weights):
        prompted_features = features + weights[0]
        support1 = torch.mm(prompted_features, weights[1])
        output1 = torch.mm(adj_norm, support1)
        support2 = torch.mm(output1, weights[2])
        output2 = torch.mm(adj_norm, support2)

        hidden = output2 @ weights[3] + weights[4]
        return hidden

    def gprompt_gnn_forward(self, features, adj_norm, weights):
        support1 = torch.mm(features, weights[1])
        output1 = torch.mm(adj_norm, support1)
        support2 = torch.mm(output1, weights[2])
        output2 = torch.mm(adj_norm, support2)
        prompted_hidden = output2 * weights[0]

        hidden = output2 @ weights[3] + weights[4]
        return hidden

    def two_views_forward(self, features, adj_norm, weights):
        weight = torch.mm(features, torch.transpose(weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < 0.5
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, weights[0])  # (n_nodes, input_dim)

        # the prompted function in "All in one" is x' = x + sum(pk)
        # and we found in surrogate module, change it to x' = x + mean(pk) is more stable in learning
        # we guess the reason behind this phenomenon is if token_num is large, then x' is far away from x at beginning
        # prompted_features = features + weighted_prompt_tokens
        prompted_features = features + 1 / self.token_num * weighted_prompt_tokens
        hidden2 = self.gnn_model(features, adj_norm)

        prompted_hidden1 = self.gnn_model(prompted_features, adj_norm)
        prompted_hidden2 = hidden2 * weights[1]

        hidden = 1 / 2 * (prompted_hidden1 + prompted_hidden2) @ weights[2] + weights[3]

        return hidden

    def sparse_all_in_one_forward(self, features, adj_norm, weights, threshold=0.5):
        assert 0.4 <= threshold <= 1, "In Sparse-All-in-one, default 0.4<= threshold <=1"

        weight = torch.mm(features, torch.transpose(weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, weights[0])  # (n_nodes, input_dim)

        # the prompted function in "All in one" is x' = x + sum(pk)
        # and we found in surrogate module, change it to x' = x + mean(pk) is more stable in learning
        # we guess the reason behind this phenomenon is if token_num is large, then x' is far away from x at beginning
        # prompted_features = features + weighted_prompt_tokens
        prompted_features = features + 1 / self.token_num * weighted_prompt_tokens

        prompted_hidden = self.gnn_model(prompted_features, adj_norm)

        hidden = prompted_hidden @ weights[1] + weights[2]
        return hidden

    def smooth_all_in_one_forward(self, features, adj_norm, weights):
        weight = torch.mm(features, torch.transpose(self.weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.softmax(weight, dim=1)
        weighted_prompt_tokens = torch.mm(weight, self.weights[0])  # (n_nodes, input_dim)

        prompted_features = features + weighted_prompt_tokens
        prompted_hidden = self.Linearized_GCN(prompted_features, adj_norm)

        hidden = prompted_hidden @ self.weights[1] + self.weights[2]
        return hidden

    def all_in_one_forward(self, features, adj_norm, weights, threshold=0.5):
        # use 'sparse all_in_one',which has adventages of speed, flexibility, Stable gradient descent process
        assert 0 <= threshold <= 1, "please set 0<=threshold<=1"

        weight = torch.mm(features, torch.transpose(weights[0], 0, 1))  # (n_nodes, token_nums)
        weight = torch.sigmoid(weight)
        mask = weight < threshold
        masked_weight = weight.masked_fill(mask, 0)  # (n_nodes, token_nums)
        weighted_prompt_tokens = torch.mm(masked_weight, weights[0])  # (n_nodes, input_dim)

        prompted_features = features + weighted_prompt_tokens

        prompted_hidden = self.gnn_model(prompted_features, adj_norm)

        hidden = prompted_hidden @ weights[1] + weights[2]
        return hidden

    def gprompt_forward(self, features, adj_norm, weights):
        embeddings = self.gnn_model(features, adj_norm)
        prompted_hidden = embeddings * weights[0]

        hidden = prompted_hidden @ weights[1] + weights[2]
        return hidden

    def gpf_forward(self, features, adj_norm, weights):
        prompted_features = features + weights[0]
        prompted_hidden = self.gnn_model(prompted_features, adj_norm)

        hidden = prompted_hidden @ weights[1] + weights[2]
        return hidden

    def gpf_plus_forward(self, features, adj_norm, weights):
        score = features @ weights[0]
        weight = F.softmax(score, dim=1)
        p = weight.mm(weights[1])

        prompted_features = features + p

        prompted_hidden = self.gnn_model(prompted_features, adj_norm)

        hidden = prompted_hidden @ weights[2] + weights[3]
        return hidden


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu