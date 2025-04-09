import numpy as np
from torch import nn, Tensor
import torch

import my_setup
from custom_layers.custom_module import BatchEvalModule


class MultiHeadAttention(BatchEvalModule):

    def __init__(self, dq, dv=None, num_heads=1, n_dq=None, n_dv=None, bias=False):
        super(MultiHeadAttention, self).__init__()

        if dv is None:
            dv = dq  # Default setup for self attention

        if n_dq is None:
            n_dq = dq / num_heads  # Default setup follows the use by Vaswani et al. 2017
        if n_dv is None:
            n_dv = dv / num_heads  # Default setup follows the use by Vaswani et al. 2017

        if num_heads < 1:
            raise Exception("We need at least a single attention head.")

        self.bias = bias
        self.num_heads = num_heads
        self.n_dv = n_dv

        self.query_transformations = nn.ModuleList([nn.Linear(in_features=dq, out_features=n_dq, bias=bias) for i in range(num_heads)])
        self.key_transformations = nn.ModuleList([nn.Linear(in_features=dq, out_features=n_dq, bias=bias) for i in range(num_heads)])
        self.value_transformations = nn.ModuleList([nn.Linear(in_features=dv, out_features=n_dv, bias=bias) for i in range(num_heads)])

        self.output_transformation = nn.Linear(in_features=num_heads * n_dv, out_features=dv, bias=bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor = None, output_attentions=False) -> Tensor:
        """
        N: batch size
        lq: query length
        lk: key and value length
        dq: query and key embedding dim
        dv: value embedding dim

        :param query: Shape is (N, ..., lq, dq)
        :param key: Shape is (N, ..., lk, dq)
        :param value: Shape is (N, ..., lk, dv)
        :param attention_mask: The positions to which to attend
        :return: The attention value
        """
        N, Lq, dq = query.size()

        attention_mask_vals = torch.zeros(size=(N, Lq, Lq), device=my_setup.DEVICE_TRAINING)
        if attention_mask is not None:
            attention_mask_vals = attention_mask_vals.masked_fill(attention_mask == 0, - torch.inf)

        head_results = [None] * self.num_heads
        attention_weights = [None] * self.num_heads
        for iN in range(self.num_heads):
            query_n = self.query_transformations[iN](query)
            key_n = self.key_transformations[iN](key)
            value_n = self.value_transformations[iN](value)

            query_key_product = query_n @ key_n.transpose(-2, -1)

            transformed_query_key_product = query_key_product / np.sqrt(query_n.size(-1))

            transformed_query_key_product += attention_mask_vals

            attention_weight = torch.softmax(transformed_query_key_product, -1)
            attention = attention_weight @ value_n

            head_results[iN] = attention

            if torch.isnan(attention).any():
                print("Produced nan value")


            attention_weights[iN] = attention_weight

        concatenated_heads = torch.cat(head_results, -1)
        output = self.output_transformation(concatenated_heads)

        if output_attentions:
            attention_weights = torch.cat([weights.unsqueeze(1) for weights in attention_weights], axis=1)
            return output, attention_weights
        return output

    @staticmethod
    def attention(self, query, key, value, attention_mask=None):

        query_key_product = query @ key.transpose(-2, -1)

        transformed_query_key_product = query_key_product / np.sqrt(query.size(-1))

        attention_weight = torch.softmax(transformed_query_key_product, -1)
        attention = attention_weight @ value

        output = attention
        if torch.isnan(output).any():
            print("Produced nan value")
        return output


if __name__ == '__main__':
    N = 8
    lq = 14
    lk = lq
    dq = 6
    dv = dq
    num_heads = 4

    multi_head_attention = MultiHeadAttention(dq, dv, n_dq=4, n_dv=4, num_heads=num_heads)
    query = torch.rand(N * lq * dq, dtype=torch.float32).reshape(N, lq, dq)

    query = torch.round(torch.rand(N * lq) * dq - 0.5).reshape(N, lq)
    query = query.to(torch.long)
    query = nn.functional.one_hot(query, num_classes=dq)
    query = query.to(torch.float32)
    key = query
    value = query

    result = multi_head_attention(query, key, value)
    print(result)

    masks = torch.round(torch.rand(N * lq))
    masks = masks.reshape(N, lq)
    masks = masks.to(torch.long)
    attention_masks = torch.logical_not(masks).unsqueeze(-1).transpose(-2, -1)
    broadcast_ones = torch.ones(size=masks.size()).unsqueeze(-1)
    attention_masks = broadcast_ones * attention_masks

    result = multi_head_attention(query, key, value, attention_mask=attention_masks)
    print(result)

    pass
