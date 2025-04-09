import torch
from torch import nn, Tensor
from torch.nn import init

import my_setup
from custom_layers.custom_module import BatchEvalModule
from custom_layers.dense import Dense


class RegressorModel(BatchEvalModule):
    def __init__(self, num_features=32, config=None):
        super(RegressorModel, self).__init__()

        default_config = {"dropout": 0.05,
                          "hidden_dim": 128}
        if config is not None:
            default_config.update(config)

        self.embedding_dim = num_features

        self.num_features_encoded_sequence = num_features * (my_setup.MAX_SEQUENCE_LENGTH + 2)

        self.num_features_regression_model = num_features

        out_features = 1 if my_setup.PREDICT_SCORE else 4

        activation = nn.GELU

        hidden_size = default_config["hidden_dim"]

        # Dropout acts as trade-off parameter between uncertainty in prediction of training data and uncertainty in the prediction of test data
        dropout_p = default_config["dropout"]

        self.mlp = nn.Sequential(
            Dense(in_features=self.num_features_regression_model, out_features=hidden_size,
                  activation=activation, dropout=None),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation, dropout=dropout_p),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation, dropout=dropout_p),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=out_features, activation=None),
        )

        pass

    def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:

        x = inputs

        x = self.mlp(x)

        output_val = x
        if my_setup.PREDICT_SCORE:

            output_val = torch.abs(x)

        else:
            output_val = torch.nn.functional.sigmoid(x)

        return output_val

    def reset(self):
        for layer in self.mlp:
            if isinstance(layer, Dense):
                layer.reset()
            elif isinstance(layer, nn.LayerNorm):
                params = list(layer.parameters())
                init.ones_(params[0])
                init.zeros_(params[1])


class RegressionModel2(BatchEvalModule):
    def __init__(self, encoder, embedding_dim=32, config=None):
        super(RegressionModel2, self).__init__()

        if config is None:
            config = {"embedding_dim": 32,
                      "dropout": 0.05,
                      "hidden_dim": 128}

        self.encoder = encoder

        self.embedding_dim = embedding_dim

        self.num_feaetures_encoded_sequence = embedding_dim * (my_setup.MAX_SEQUENCE_LENGTH + 2)

        self.num_features_regression_model = embedding_dim

        self.flatten = nn.Flatten()

        out_features = 1 if my_setup.PREDICT_SCORE else 4

        activation = nn.GELU

        hidden_size = 128 if "hidden_dim" not in config else config["hidden_dim"]
        dropout_p = 0.01 if "dropout" not in config else config["dropout"]

        self.mlp = nn.Sequential(
            Dense(in_features=self.num_features_regression_model, out_features=hidden_size,
                  activation=activation, dropout=None),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation, dropout=dropout_p),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation, dropout=dropout_p),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            Dense(in_features=hidden_size, out_features=hidden_size, activation=activation),
            nn.LayerNorm(hidden_size),
            Dense(in_features=hidden_size, out_features=out_features, activation=None),
        )

        pass

    def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:
        encoded_input_seq = self.encoder(inputs)

        encoded_input_seq = encoded_input_seq[:, 0]
        seq_features = self.flatten(encoded_input_seq)
        x = seq_features

        x = self.mlp(x)

        output_val = x
        if my_setup.PREDICT_SCORE:
            output_val = torch.abs(x)
        else:
            output_val = torch.nn.functional.sigmoid(x)

        return output_val
