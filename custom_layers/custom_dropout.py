from typing import TypeVar

import torch
from torch import Tensor

import my_setup
from custom_layers.custom_module import BatchEvalModule

T = TypeVar('T', bound='Module')


class Dropout(BatchEvalModule):
    # A dropout layer following the idea of Tensorflow
    def __init__(self, p=0):
        super(Dropout, self).__init__()

        self.p = p

        self.scale_factor = 1 / (1 - p)

        pass


    def forward(self, x: Tensor) -> Tensor:
        in_val = x

        if self.batch_evaluation:
            batch_size = x.size(0)
            dropout_positions = torch.bernoulli(x[0], self.p)
            dropout_positions = dropout_positions.repeat(batch_size, *([1] * (len(x.size()) - 1)))
        elif self.training:
            dropout_positions = torch.bernoulli(x, self.p)
            dropout_positions.to(my_setup.DEVICE_TRAINING)


        if self.batch_evaluation or self.training:
            x = torch.zeros(x.size(), device=my_setup.DEVICE_TRAINING)
            x[dropout_positions == 0] = in_val[dropout_positions == 0]
            x *= self.scale_factor

        return x


if __name__ == '__main__':
    m = Dropout(0.5)

    a = torch.arange(150).reshape(3, 5, -1) + 1.0

    print("Training")
    m.train(True)
    res = m(a)
    print(res)

    print("Eval")
    m.eval()
    res = m(a)
    print(res)

    print("Batch Eval")
    m.batch_eval()
    res = m(a)
    print(res)

    print("Training 2")
    m.train(True)
    res = m(a)
    print(res)

    print("Batch Eval 2")
    m.batch_eval()
    res = m(a)
    print(res)

    print("Eval 2")
    m.eval()
    res = m(a)
    print(res)