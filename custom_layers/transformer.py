from torch import nn, Tensor

from custom_layers.custom_module import BatchEvalModule
from custom_layers.dense import Dense
from custom_layers.multihead_attention import MultiHeadAttention


# Implements a transformer encoder as in "Attention is all you need" Vaswani et al. 2017
# Combination of Self-Attention and per Position Feed Forward network
class TransformerEncoder(BatchEvalModule):
    def __init__(self, num_heads, d_embedding, d_attention, d_hidden=None, activation=None, dropout=None):
        super(TransformerEncoder, self).__init__()

        if d_hidden is None:
            d_hidden = d_embedding

        if activation is None:
            activation = nn.GELU

        self.self_attention = MultiHeadAttention(dq=d_embedding, dv=d_embedding, n_dq=d_attention, n_dv=d_attention,
                                                 num_heads=num_heads)
        self.layernorm_1 = nn.LayerNorm(d_embedding)
        self.ffn = nn.Sequential(
            Dense(in_features=d_embedding, out_features=d_hidden, activation=activation, dropout=dropout),
            Dense(in_features=d_hidden, out_features=d_embedding, activation=None, dropout=dropout)
        )
        self.layernorm_2 = nn.LayerNorm(d_embedding)

        pass

    def forward(self, input_tensor: Tensor, attention_mask: Tensor = None, output_attentions=True) -> Tensor:
        attention_input = input_tensor

        x = attention_input
        res = self.self_attention(x, x, x, attention_mask=attention_mask, output_attentions=output_attentions)
        self_attention = res
        if output_attentions:
            self_attention, attention_weights = res

        x = self_attention + attention_input  # Residual connection
        x = self.layernorm_1(x)
        ffn_input = x
        ffn_output = self.ffn(x)
        x = ffn_output + ffn_input
        x = self.layernorm_2(x)

        output_tensor = x

        if output_attentions:
            return output_tensor, attention_weights
        return output_tensor
