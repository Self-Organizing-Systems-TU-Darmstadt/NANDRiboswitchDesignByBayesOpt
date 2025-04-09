import torch
from torch import nn, Tensor


class CustomEmbedding(nn.Module):
    # A custom embedding class to circumvent error of default implementation.



    def __init__(self, num_classes, embedding_dim):
        super(CustomEmbedding, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(in_features=num_classes, out_features=embedding_dim)
        pass

    def forward(self, input_val: Tensor) -> Tensor:
        # Simple embedding equals one hot encoding followed by a linear transformation

        x = input_val

        x = x.to(torch.int64)
        x = nn.functional.one_hot(x, num_classes=self.num_classes)
        x = x.to(torch.float)

        x = self.linear(x)

        output_val = x
        return output_val
