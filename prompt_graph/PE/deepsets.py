import torch
from torch import nn


class DeepSets(nn.Module):
    layers: nn.ModuleList

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, activation: str='relu') -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = DeepSetsLayer(in_dims, hidden_dims, activation)
            self.layers.append(layer)
            in_dims = hidden_dims
        # layer = DeepSetsLayer(hidden_dims, out_dims, activation='id') # drop last activation
        layer = DeepSetsLayer(hidden_dims, out_dims, activation)
        self.layers.append(layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [N, D_in]
        :return: Output feature matrix. [N, D_out]
        """
        for layer in self.layers:
            X = layer(X)   # [N, D_out]
        return X           # [N, D_out]


class DeepSetsLayer(nn.Module):
    fc_one: nn.Linear
    fc_all: nn.Linear
    activation: nn.Module

    def __init__(self, in_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()
        self.fc_curr = nn.Linear(in_dims, out_dims, bias=True)
        self.fc_all = nn.Linear(in_dims, out_dims, bias=False)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [N, D_in]
        :return: Output feature matrix. [N, D_out]
        """
        Z_curr = self.fc_curr(X)                          # [N, D_out]
        Z_all = self.fc_all(X.sum(dim=0, keepdim=True))   # [1, D_out]
        # Z_all = self.fc_all(X.max(dim=1, keepdim=True)[0])   # [1, D_out]
        X = Z_curr + Z_all                                # [N, D_out]
        return self.activation(X)                         # [N, D_out]

