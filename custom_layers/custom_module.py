from typing import TypeVar

from torch import nn

T = TypeVar('T', bound='Module')


class BatchEvalModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BatchEvalModule, self).__init__(*args, **kwargs)

        self.batch_evaluation = False

    def batch_eval(self):
        def set_batch_eval(module):
            module.eval()
            if isinstance(module, BatchEvalModule):
                module.batch_eval()
            elif hasattr(module, "children"):
                [set_batch_eval(child_mod) for child_mod in module.children()]

        self.eval()
        self.batch_evaluation = True

        for module in self.children():
            set_batch_eval(module)

    def train(self: T, mode: bool = True) -> T:
        """
        This method wraps the original train by accounting for the additionally introduced mode batch eval.
        This equals eval mode except for the custom_dropout.Dropout, which then applies batch wise dropout.
        This means, that the same dropout is applied to all values in a batch.

        :param mode: True for training (activating dropout etc.) and False for evaluation
        :return:
        """
        self.batch_evaluation = False

        return super().train(mode)
