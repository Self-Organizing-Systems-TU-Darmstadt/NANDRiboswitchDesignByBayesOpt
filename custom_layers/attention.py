import numpy as np
from torch import nn, Tensor
import torch


class Attention(nn.Module):

    def __init__(self, ed, ev, n_ed=None, n_ev=None):
        super().__init__()

        if n_ed is None:
            n_ed = ed
        if n_ev is None:
            n_ev = ev

        self.query_weight = nn.Parameter(torch.eye(ed, n_ed))
        self.key_weight = nn.Parameter(torch.eye(ed, n_ed))
        self.value_weight = nn.Parameter(torch.eye(ev, n_ev))

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor = None) -> Tensor:
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



        query_n = query @ self.query_weight
        key_n = key @ self.key_weight
        value_n = value @ self.value_weight

        query_key_product = query_n @ key_n.transpose(-2, -1)


        transformed_query_key_product = query_key_product / np.sqrt(query_n.size(-1))
        if attention_mask is not None:
            attention_mask_vals = torch.zeros(size=attention_mask.size())
            attention_mask_vals = attention_mask_vals.masked_fill(attention_mask == 0, - torch.inf)
            transformed_query_key_product += attention_mask_vals

        attention_weight = torch.softmax(transformed_query_key_product, -1)
        attention = attention_weight @ value_n

        output = attention
        if torch.isnan(output).any():
            print("Produced nan value")
        return output


if __name__ == '__main__':
    N = 8
    lq = 14
    lk = lq
    ed = 6
    ev = ed

    attention = Attention(ed, ev, n_ed=ed * 2, n_ev=ev)
    query = torch.rand(N * lq * ed, dtype=torch.float32).reshape(N, lq, ed)

    query = torch.round(torch.rand(N * lq) * ed - 0.5).reshape(N, lq)
    query = query.to(torch.long)
    query = nn.functional.one_hot(query, num_classes=ed)
    query = query.to(torch.float32)
    key = query
    value = query

    result = attention(query, key, value)
    print(result)

    pass
