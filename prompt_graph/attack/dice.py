import torch
from prompt_graph.attack import PreTrain_task
import torch.nn as nn
from prompt_graph.data import load4node, load4graph, load4node_to_sparse, split_train_val_test
#from prompt_graph.prompt import LightPrompt_token, Gprompt
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
import random


class DICE(BaseAttack):
    def __init__(self, budget:float=0.05, *arg, **kwargs):
        super(DICE, self).__init__(*arg, **kwargs)
        if self.undirected:
            self.n_perturbations = int(self.nedges * budget //2)
        else:
            self.n_perturbations = int(self.nedges * budget)


    def attack(self):
        modified_adj = self.adj_ori.clone()

        remove_or_insert = np.random.choice(2, self.n_perturbations)
        n_remove = sum(remove_or_insert)
        indices = modified_adj.nonzero()
        possible_indices = [x for x in zip(indices[:, 0], indices[:, 1])
                            if self.y[x[0]] == self.y[x[1]]]

        remove_indices = random.sample(possible_indices, n_remove)
        remove_indices_0, remove_indices_1 = [x[0] for x in remove_indices], [x[1] for x in remove_indices]
        modified_adj[remove_indices_0, remove_indices_1] = 0
        if self.undirected:
            modified_adj[remove_indices_1, remove_indices_0] = 0

        n_insert = self.n_perturbations - n_remove
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(modified_adj.shape[0], n_remaining),
                                        np.random.choice(modified_adj.shape[0], n_remaining)]).T

            # filter out existing edges, and pairs with the different labels
            candidate_edges = set([(u, v) for u, v in candidate_edges if self.y[u] != self.y[v]
                                        and modified_adj[u, v] == 0 and modified_adj[v, u] == 0])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            if self.undirected:
                modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.modified_adj = modified_adj.detach() if self.attack_structure else None
        self.modified_features = None