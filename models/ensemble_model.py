import torch
from torch import Tensor, nn
from torch.nn import init

from custom_layers.custom_module import BatchEvalModule


class EnsembleModel2(BatchEvalModule):
    def __init__(self, encoder, model_class, model_params=None, ensemble_size=10):
        super(EnsembleModel2, self).__init__()

        self.encoder = encoder
        self.ensemble_size = ensemble_size

        self.flatten = nn.Flatten()

        # Instantiate all the models in the ensemble
        models = [model_class(**model_params) for iM in range(ensemble_size)]
        self.models = nn.ModuleList(modules=models)

        pass

    def forward(self, inputs: Tensor, combine_outputs=True) -> Tensor:
        encoded_input_seq = self.encoder(inputs)

        encoded_input_seq = encoded_input_seq[:, 0]

        seq_features = self.flatten(encoded_input_seq)


        model_outputs = [None] * self.ensemble_size
        for iM, model in enumerate(self.models):
            cur_outputs = model(seq_features)
            model_outputs[iM] = cur_outputs.unsqueeze(-1)  # Adds an additional dimension for the ensemble.

        outputs = torch.cat(model_outputs, dim=-1)

        if combine_outputs:
            outputs = torch.mean(outputs, dim=-1)
        return outputs

    def reset(self, reset_encoder=False):
        print(f"Resetting Params (reset_encoder={reset_encoder})")
        if reset_encoder:
            if hasattr(self.encoder, "reset"):
                self.encoder.reset()
            else:
                raise Exception("Encoder has no attribute 'reset'.")


        for model in self.models:
            model.reset()

        pass


class EnsembleModel(BatchEvalModule):
    def __init__(self, model_class, model_params=None, ensemble_size=10):
        super(EnsembleModel, self).__init__()

        self.ensemble_size = ensemble_size
        # Instantiate all the models in the ensemble
        models = [model_class(**model_params) for iM in range(ensemble_size)]

        # ##########################################################################################
        # in_val = torch.LongTensor([[2, 30, 43, 31, 47, 47, 48, 52, 68, 68, 66, 60, 36, 3, 0, 0],
        #                            [2, 39, 16, 50, 59, 32, 52, 67, 64, 50, 60, 3, 0, 0, 0, 0]])
        # models[0].eval()
        # test_output = models[0].encoder(in_val)
        # print("models[0].encoder(in_val):\n", test_output[:, 0, :6].cpu().detach().numpy())
        # ##########################################################################################

        self.models = nn.ModuleList(modules=models)

        # ##########################################################################################
        # in_val = torch.LongTensor([[2, 30, 43, 31, 47, 47, 48, 52, 68, 68, 66, 60, 36, 3, 0, 0],
        #                            [2, 39, 16, 50, 59, 32, 52, 67, 64, 50, 60, 3, 0, 0, 0, 0]])
        # self.eval()
        # test_output = self.models[0].encoder(in_val)
        # print("self.models[0].encoder(in_val):\n", test_output[:, 0, :6].cpu().detach().numpy())
        # ##########################################################################################

        ### self.reset() # This would also reset the parameters of the encoder model in the current implementation

        # ##########################################################################################
        # in_val = torch.LongTensor([[2, 30, 43, 31, 47, 47, 48, 52, 68, 68, 66, 60, 36, 3, 0, 0],
        #                            [2, 39, 16, 50, 59, 32, 52, 67, 64, 50, 60, 3, 0, 0, 0, 0]])
        # self.eval()
        # test_output = self.models[0].encoder(in_val)
        # print("self.models[0].encoder(in_val):\n", test_output[:, 0, :6].cpu().detach().numpy())
        # ##########################################################################################
        pass

    def forward(self, inputs: Tensor, combine_outputs=True) -> Tensor:
        model_outputs = [None] * self.ensemble_size
        for iM, model in enumerate(self.models):
            cur_outputs = model(inputs)
            model_outputs[iM] = cur_outputs.unsqueeze(-1)  # Adds an additional dimension for the ensemble.

        # outputs = torch.stack(model_outputs).transpose(-1, 0)
        outputs = torch.cat(model_outputs, dim=-1)

        if combine_outputs:
            outputs = torch.mean(outputs, dim=-1)
        return outputs

    def reset(self):
        # raise Exception("Currently reset, resets all parameters. This includes the one of the pretrained encoder model.")
        print("Currently reset, resets all parameters. This includes the one of the pretrained encoder model.")
        for entry in self.parameters():
            init.normal_(entry)
        pass
