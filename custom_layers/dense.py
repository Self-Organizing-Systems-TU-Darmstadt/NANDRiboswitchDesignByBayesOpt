import torch.nn as nn
from torch import Tensor
from torch.nn import init

from custom_layers.custom_dropout import Dropout
from custom_layers.custom_module import BatchEvalModule


class Dense(BatchEvalModule):
    # A dense layer following the idea of Tensorflow
    def __init__(self, in_features, out_features, activation=None, dropout: float = None):
        super().__init__()

        self.dropout = dropout
        if dropout is not None:
            self.dropout = Dropout(p=dropout)
            # Implement batch_eval functionality

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.activation = activation

        # Use get implementation by name for the activation function
        if activation:
            self.activation = activation()

        self.reset()
        # self.linear.weight = nn.Parameter(torch.eye(6) * 10)

        # init.xavier_normal_(self.linear.weight)
        # init.zeros_(self.linear.bias)
        # self.linear.weight = nn.Parameter(torch.empty(self.linear.weight.shape).bernoulli(p=0.1) * 2 - 1)
        # self.linear.weight = nn.Parameter(torch.empty(self.linear.weight.shape).uniform_(-1, 1))

        # init.normal_(self.linear.weight, std=0.1)
        # init.dirac_(self.linear.weight)
        # init.normal_(self.linear.bias, std=0.5)

        # init.eye_(self.linear.weight)
        # init.zeros_(self.linear.bias)

        pass

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def reset(self):
        init.xavier_normal_(self.linear.weight)
        # init.normal(self.linear.weight)
        init.zeros_(self.linear.bias)
        # init.xavier_normal_(self.linear.bias)

