import numpy as np
import torch
import torch.nn as nn

import my_setup
from torch import Tensor

from custom_layers.attention import Attention
from custom_layers.dense import Dense
from custom_layers.multihead_attention import MultiHeadAttention
from custom_layers.transformer import TransformerEncoder


class TransformerModel2(nn.Module):
    def __init__(self, max_sequence_length=128, vocabulary_size=128, embedding_dim=16, dropout=0.1):
        super(TransformerModel2, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        max_input_length = max_sequence_length + 2

        X = np.arange(max_input_length).repeat(embedding_dim).reshape(-1, embedding_dim)
        d = np.arange(embedding_dim) + 1

        X = X / d + d
        pe = np.cos(X * np.pi)
        pe = torch.Tensor(pe)
        self.positional_encoding = nn.Parameter(pe, requires_grad=False)

        num_heads = 4

        d_attention = int(np.ceil(embedding_dim / num_heads))
        d_attention *= 2  # Extended Attention dimension in comparison to Attention is all you need.

        activation = nn.GELU
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(num_heads,
                                   d_embedding=embedding_dim,
                                   d_attention=d_attention,
                                   d_hidden=3 * embedding_dim,
                                   activation=activation,
                                   dropout=dropout)
                for _ in range(6)
            ])

        pass

    def forward(self, input_val: Tensor, masks: Tensor = None, output_attentions=False) -> Tensor:
        # attention_masks = None
        attention_masks = torch.ones(input_val.size(), device=my_setup.DEVICE_TRAINING)
        attention_masks[input_val == my_setup.TOKENIZER.PAD_TOKEN_ID] = 0
        if masks is not None:
            attention_masks *= torch.logical_not(masks)
            pass

        attention_masks = attention_masks.unsqueeze(-1).transpose(-2, -1)
        broadcast_ones = torch.ones(size=input_val.size(), device=my_setup.DEVICE_TRAINING).unsqueeze(-1)
        attention_masks = broadcast_ones * attention_masks

        batch_size, seq_length = input_val.size()
        x = input_val
        x = self.embedding(x)
        x += self.positional_encoding[:seq_length]
        attentions = []
        for transformer in self.transformers:
            res = transformer(x, attention_masks, output_attentions=output_attentions)
            if output_attentions:
                x, attention_weights = res
                attentions.append(attention_weights)
            else:
                x = res

        output_val = x
        if output_attentions:
            attentions = torch.cat([weights.unsqueeze(1) for weights in attentions], axis=1)
            return output_val, attentions
        return output_val


class TransformerModel(nn.Module):
    # def __init__(self, max_sequence_length=128, vocabulary_size=128, embedding_dim=4):
    def __init__(self, max_sequence_length=14, vocabulary_size=128, embedding_dim=4):
        super(TransformerModel, self).__init__()

        vocab_size = my_setup.VOCABULARY_SIZE
        special_characters_vocab_size = my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE

        n_input_vals = vocab_size + special_characters_vocab_size + 4

        # n_input_vals = embedding_dim
        n_sequential_out_vals = max_sequence_length * embedding_dim
        n_out_vals = 5

        scale_factor = 1

        self.n_input_vals = n_input_vals

        # self.embedding_noise_and_positional_encoding_layer = EmbeddingAndNoiseAndPositionalEncoding(
        #     vocab_size=vocab_size + my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE,
        #     embedding_dim=embedding_dim,
        #     dropout=None,
        #     max_len=sequence_length,
        #     std_dev=None)

        positional_encoding = torch.empty(size=(max_sequence_length, 2))
        positional_encoding[:, 0] = (torch.arange(max_sequence_length) - max_sequence_length / 2 + 0.5) / (
                max_sequence_length / 2 - 0.5)
        positional_encoding[:, 1] = torch.sign(positional_encoding[:, 0])
        positional_encoding[:, 0] = torch.abs(positional_encoding[:, 0])
        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=False)

        num_heads = 2

        d_attention = 8

        self.transformers = nn.ModuleList(
            [TransformerEncoder(num_heads, d_embedding=n_input_vals, d_attention=d_attention, activation=nn.GELU,
                                dropout=0.2)
             for iX in range(8)])

        self.flatten = nn.Flatten()
        self.linear = Dense(in_features=n_input_vals * max_sequence_length,
                            out_features=3 * max_sequence_length, dropout=0.2)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(max_sequence_length, 3))
        # self.linear = Dense(in_features=n_input_vals, out_features=3)

        # self.flatten = nn.Flatten()
        # self.mlp = nn.Sequential(Dense(in_features=3 * max_sequence_length, out_features=n_out_vals*max_sequence_length, activation=nn.GELU),
        #                         Dense(in_features=n_out_vals * max_sequence_length, out_features=n_out_vals * max_sequence_length))
        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(max_sequence_length, n_out_vals))
        self.mlp = nn.Sequential(Dense(in_features=3, out_features=n_out_vals, activation=nn.GELU),
                                 Dense(in_features=n_out_vals, out_features=n_out_vals))
        pass

    def forward(self, input_val: Tensor, masks: Tensor = None) -> Tensor:
        attention_masks = None
        if masks is not None:
            attention_masks = torch.logical_not(masks).unsqueeze(-1).transpose(-2, -1)
            broadcast_ones = torch.ones(size=masks.size()).unsqueeze(-1)
            attention_masks = broadcast_ones * attention_masks
            pass

        x = input_val
        x = x.to(torch.int64)
        x = nn.functional.one_hot(x, num_classes=self.n_input_vals)
        # x = self.embedding_noise_and_positional_encoding_layer(x)
        x = x.to(torch.float)
        x[:, :, -2:] += self.positional_encoding

        for transformer in self.transformers:
            x = transformer(x, attention_masks)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.unflatten(x)

        # x = self.flatten(x)
        x = self.mlp(x)
        # x = self.unflatten(x)
        # output_val = torch.transpose(x, 1, 2)
        output_val = x
        return output_val


class TokenLevelClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TokenLevelClassifier, self).__init__()

        self.dense1 = Dense(in_features=input_dim, out_features=output_dim,
                            activation=nn.GELU, dropout=0.2)

        # self.flatten = nn.Flatten()
        # self.mlp = Dense(in_features=output_dim * my_setup.MAX_SEQUENCE_LENGTH,
        #                  out_features=output_dim * my_setup.MAX_SEQUENCE_LENGTH,
        #                  activation=nn.GELU, dropout=0.2)
        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(my_setup.MAX_SEQUENCE_LENGTH, output_dim))

        self.dense2 = Dense(in_features=output_dim, out_features=output_dim,
                            activation=None, dropout=None)

    def forward(self, input_val: Tensor) -> Tensor:
        x = input_val
        x = self.dense1(x)

        # x = self.flatten(x)
        # x = self.mlp(x)
        # x = self.unflatten(x)

        x = self.dense2(x)
        # x = torch.transpose(x, -1, -2)
        output_val = x
        return output_val


class SimpEncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SimpEncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_val: Tensor, masks: Tensor = None, output_encoding=False) -> Tensor:
        encoded_val = self.encoder(input_val, masks)
        decoded_val = self.decoder(encoded_val)

        # Alternative could be to apply a softmax along the embedding dimension to yield a vector normalized to 1.
        if output_encoding:
            # Turn the encoding into length 1 vectors.
            # encoding = encoded_val / torch.sqrt(torch.sum(torch.square(encoded_val), dim=-1).unsqueeze(-1).repeat(1, 1, 128))
            encoding = encoded_val
            return decoded_val, encoding

        return decoded_val


class SimpleModel2(nn.Module):
    def __init__(self, sequence_length=14, embedding_dim=4):
        super(SimpleModel2, self).__init__()

        embedding_dim = 5

        vocab_size = my_setup.VOCABULARY_SIZE
        special_characters_vocab_size = my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE

        n_input_vals = vocab_size + special_characters_vocab_size + 25

        # n_input_vals = embedding_dim
        n_sequential_out_vals = sequence_length * embedding_dim
        n_out_vals = vocab_size

        scale_factor = 1

        self.n_input_vals = n_input_vals

        # self.embedding_noise_and_positional_encoding_layer = EmbeddingAndNoiseAndPositionalEncoding(
        #     vocab_size=vocab_size + my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE,
        #     embedding_dim=embedding_dim,
        #     dropout=None,
        #     max_len=sequence_length,
        #     std_dev=None)

        positional_encoding = torch.empty(size=(sequence_length, 2))
        positional_encoding[:, 0] = (torch.arange(sequence_length) - sequence_length / 2 + 0.5) / (
                sequence_length / 2 - 0.5)
        positional_encoding[:, 1] = torch.sign(positional_encoding[:, 0])
        positional_encoding[:, 0] = torch.abs(positional_encoding[:, 0])
        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=False)

        num_heads = 2

        self.attention1 = Attention(ed=n_input_vals, ev=n_input_vals, n_ed=n_input_vals, n_ev=n_input_vals)
        self.attention2 = Attention(ed=n_input_vals, ev=n_input_vals, n_ed=n_input_vals, n_ev=n_input_vals)
        self.attention3 = Attention(ed=n_input_vals, ev=n_input_vals, n_ed=n_input_vals, n_ev=n_input_vals)

        n_dq = 16
        n_dv = n_dq
        self.attention1 = MultiHeadAttention(dq=n_input_vals, dv=n_input_vals, n_dq=n_dq, n_dv=n_dv,
                                             num_heads=num_heads)
        self.attention2 = MultiHeadAttention(dq=n_input_vals, dv=n_input_vals, n_dq=n_dq, n_dv=n_dv,
                                             num_heads=num_heads)
        self.attention3 = MultiHeadAttention(dq=n_input_vals, dv=n_input_vals, n_dq=n_dq, n_dv=n_dv,
                                             num_heads=num_heads)

        self.layer_norm1 = nn.LayerNorm(n_input_vals)
        self.layer_norm2 = nn.LayerNorm(n_input_vals)
        self.layer_norm3 = nn.LayerNorm(n_input_vals)

        self.net1 = nn.Sequential(Dense(in_features=n_input_vals,
                                        out_features=n_input_vals * 2,
                                        activation=nn.GELU),
                                  # Dense(in_features=n_input_vals, out_features=n_input_vals, activation=nn.GELU),
                                  # Dense(in_features=n_input_vals, out_features=n_input_vals, activation=nn.GELU),
                                  Dense(in_features=n_input_vals * 2,
                                        out_features=n_input_vals * 2,
                                        activation=nn.GELU),
                                  Dense(in_features=n_input_vals * 2,
                                        out_features=n_input_vals,
                                        activation=nn.GELU),
                                  nn.LayerNorm(n_input_vals)
                                  )

        self.net2 = nn.Sequential(Dense(in_features=n_input_vals,
                                        out_features=n_input_vals,
                                        activation=nn.GELU),
                                  # Dense(in_features=n_input_vals, out_features=n_input_vals, activation=nn.GELU),
                                  # Dense(in_features=n_input_vals, out_features=n_input_vals, activation=nn.GELU),
                                  Dense(in_features=n_input_vals,
                                        out_features=n_input_vals,
                                        activation=nn.GELU),
                                  Dense(in_features=n_input_vals,
                                        out_features=n_input_vals,
                                        activation=nn.GELU),
                                  nn.LayerNorm(n_input_vals)
                                  )

        self.linear = Dense(in_features=n_input_vals,
                            out_features=n_out_vals)

        pass

    def forward(self, input_val: Tensor, masks: Tensor = None) -> Tensor:
        attention_masks = None
        if masks is not None:
            attention_masks = torch.logical_not(masks).unsqueeze(-1).transpose(-2, -1)
            broadcast_ones = torch.ones(size=masks.size()).unsqueeze(-1)
            attention_masks = broadcast_ones * attention_masks
            pass

        x = input_val
        x = x.to(torch.int64)
        x = nn.functional.one_hot(x, num_classes=self.n_input_vals)
        # x = self.embedding_noise_and_positional_encoding_layer(x)
        x = x.to(torch.float)
        x[:, :, -2:] += self.positional_encoding
        x = self.attention1(x, x, x, attention_mask=attention_masks)
        x = self.layer_norm1(x)
        x = self.net1(x)
        x = self.attention2(x, x, x, attention_mask=attention_masks)
        x = self.layer_norm2(x)
        x = self.net2(x)
        # x = self.attention3(x, x, x)
        # x = self.layer_norm3(x)
        x = self.linear(x)
        output_val = torch.transpose(x, 1, 2)
        return output_val


class SimpleModel(nn.Module):
    def __init__(self, sequence_length=14, embedding_dim=4):
        super(SimpleModel, self).__init__()

        vocab_size = my_setup.VOCABULARY_SIZE
        special_characters_vocab_size = my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE

        n_input_vals = sequence_length * embedding_dim
        n_sequential_out_vals = sequence_length * embedding_dim
        n_out_vals = sequence_length * vocab_size

        scale_factor = 1

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size + special_characters_vocab_size,
                                            embedding_dim=embedding_dim)

        self.flatten = nn.Flatten()

        self.sequential = nn.Sequential(
            Dense(in_features=n_input_vals, out_features=n_input_vals * scale_factor, activation=nn.GELU),
            nn.LayerNorm(n_input_vals * scale_factor),
            # Dense(in_features=n_input_vals * scale_factor, out_features=n_input_vals * scale_factor, activation=nn.GELU, dropout=0.2),
            # nn.LayerNorm(n_input_vals * scale_factor),
            # Dense(in_features=n_input_vals * scale_factor, out_features=n_input_vals * scale_factor, activation=nn.GELU, dropout=0.2),
            # nn.LayerNorm(n_input_vals * scale_factor),
            Dense(in_features=n_input_vals * scale_factor, out_features=n_input_vals, activation=nn.GELU, dropout=0.2),
            nn.LayerNorm(n_input_vals),
            Dense(in_features=n_input_vals, out_features=n_input_vals, activation=nn.GELU, dropout=0.2),
            nn.LayerNorm(n_input_vals),
            Dense(in_features=n_input_vals, out_features=n_sequential_out_vals, activation=nn.GELU),
        )

        self.linear = Dense(in_features=n_sequential_out_vals,
                            out_features=n_out_vals,
                            activation=None)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(vocab_size, sequence_length))
        pass

    def forward(self, input_val: Tensor) -> Tensor:
        x = input_val
        # x = self.embedding_layer(x)
        x = x.to(torch.int64)
        x = nn.functional.one_hot(x,
                                  num_classes=self.vocab_size + my_setup.SPECIAL_CHARACTERS_VOCAB_SIZE)
        x = x.to(torch.float)
        # x = torch.reshape(x, [-1, self.sequence_length * self.embedding_dim])
        x = self.flatten(x)

        x = self.sequential(x)

        x = self.linear(x)

        output_val = self.unflatten(x)
        # output_val = torch.reshape(x, (-1, self.vocab_size, self.sequence_length))
        return output_val

    def get_gradients(self):
        layers = [self.dense1, self.dense3, self.linear]
        gradients = []
        for layer in layers:
            cur_gradients = []
            for param in layer.parameters():
                if param.requires_grad and param.grad is not None:
                    cur_gradients.append(param.grad.view(-1))
            gradients.append(cur_gradients)

        return gradients
