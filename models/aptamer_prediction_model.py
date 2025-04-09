import torch.multiprocessing as multiprocessing
import concurrent.futures

import os
import shutil

import time
import numpy as np
import pandas as pd
import torch
import yaml
# Ignite requires version 0.4.12 or 0.4.13 to allow for "save_as_state_dict" option.
from ignite.contrib.metrics.regression import MeanError, MaximumAbsoluteError, R2Score
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, create_lr_scheduler_with_warmup
from ignite.handlers import Timer
from ignite.metrics import MeanSquaredError, MeanAbsoluteError, Loss
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.utils.data import DataLoader

import my_setup
from data_processing.masked_sequence_model_token_based_triplet_dataset import Sequences
from data_processing.sequence_to_expression_dataset import SequenceToExpressionDataset, Measurements
from loss_function.regression_loss import bernoulli_kl_divergence
from models.simple_model import TransformerModel2
from models.simple_regression_models import RegressionModel2, RegressorModel
from utils.my_utils import get_best_params_path


class JointModel(nn.Module):
    def __init__(self, config, id=-1):
        super(JointModel, self).__init__()
        self.is_trained = False
        self.config = config
        self.device = config["training_regression"]["device"]
        self.predict_score = config["training_regression"]["predict_score"]
        self.embedding_dim = config['encoder']['embedding_dim']
        self.max_sequence_length = config['general']['max_sequence_length']
        self.model_settings_id = f"dim={self.embedding_dim}_max-len={self.max_sequence_length}"
        self.tokenizer = my_setup.TOKENIZER

        self.optimizer = None

        self.id = id

        # Load the model here
        self.load_models()
        print("Loaded Models")

        # Reset the model
        pass

        if self.predict_score:
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = bernoulli_kl_divergence

    def forward(self, inputs: Tensor, output_attentions=False,  *args, **kwargs) -> Tensor:

        res = self.encoder(inputs, output_attentions=output_attentions)
        latent_representation = res
        if output_attentions:
            latent_representation, attentions = res

        sequence_embeddings = latent_representation[:, 0]
        outputs = self.regressor(sequence_embeddings)

        if output_attentions:
            return outputs, attentions
        return outputs

    def load_models(self):
        self.load_encoder()
        self.load_regressor()

    def reset(self):
        self.reset_encoder()
        self.reset_regressor()

    def reset_encoder(self):

        encoder_parameters_path = self.config["encoder"]["params_path"]

        if encoder_parameters_path is None or not os.path.exists(encoder_parameters_path):
            print(
                f"The required parameters of the pretrained encoder do not exist at {encoder_parameters_path} (model_settings_id={self.model_settings_id}).")
            return

        print("Loading Encoder Parameters from:", encoder_parameters_path)
        state_dict = torch.load(encoder_parameters_path, map_location=self.device)
        is_whole_model_state_dict = any([key.startswith("encoder.") for key in state_dict])
        if is_whole_model_state_dict:
            state_dict = {key.replace("encoder.", ""): state_dict[key] for key in state_dict if
                          key.startswith("encoder.")}
        self.encoder.load_state_dict(state_dict=state_dict)

        print("Loaded Encoder Parameters from:", encoder_parameters_path)

    def reset_regressor(self):
        self.regressor.reset()

    def load_encoder(self):
        config = self.config
        embedding_dim = config["encoder"]["embedding_dim"]
        dropout = config["encoder"]["dropout"]

        encoder = TransformerModel2(max_sequence_length=my_setup.MAX_SEQUENCE_LENGTH,
                                    vocabulary_size=my_setup.TOKENIZER.VOCABULARY_SIZE,
                                    embedding_dim=embedding_dim,
                                    dropout=dropout)

        # short_summary(encoder)

        encoder = encoder.to(self.device)

        print("Moved Encoder to:", self.device)
        self.encoder = encoder
        self.reset_encoder()

    def load_regressor(self):
        regressor = RegressorModel(num_features=self.config["encoder"]["embedding_dim"],
                                   config=self.config["regression"])
        # short_summary(regressor)
        regressor = regressor.to(self.device)
        print("Moved Regressor To:", self.device)
        self.regressor = regressor

        self.reset_regressor()

    def train_step(self, engine, batch):
        """
        One can use dropout in the encoder module.
        As Dropout mitigates overfitting, the uncertainty difference between training and test samples is reduced.
        One suitable adaption could be using a smaller amount of dropout!
        """

        self.train(True)

        self.optimizer.zero_grad()
        inputs, target_outputs = batch
        inputs = inputs.to(self.device)
        target_outputs = target_outputs.to(self.device)
        model_outputs = self(inputs, combine_outputs=False)

        loss = self.loss_func(model_outputs, target_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, engine, batch):

        self.eval()
        with torch.no_grad():
            self.optimizer.zero_grad()
            inputs, target_outputs = batch
            inputs = inputs.to(self.device)
            target_outputs = target_outputs.to(self.device)
            model_outputs = self(inputs)

            model_outputs = model_outputs.flatten()
            target_outputs = target_outputs.flatten()
            return model_outputs, target_outputs  # Output needs to be of the form y_pred, y_true

    def _train_model(self, fit_config, train_loader, val_loader=None):
        learning_rate = float(fit_config["learning_rate"])

        train_embedding = bool(self.config["training_regression"]["train_encoder"])

        if train_embedding:
            parameters = self.parameters()
        else:
            parameters = self.regressor.parameters()

        parameters = list(parameters)
        optimizer = Adam(params=parameters, lr=learning_rate, weight_decay=fit_config["weight_decay"])
        self.optimizer = optimizer
        loss_func = self.loss_func
        print("Num Parameters to train:",
              sum(list(map(lambda param: param.numel() if param.requires_grad else 0, parameters))))

        trainer = Engine(self.train_step)
        evaluator = Engine(self.eval_step)

        # Learning Rate Scheduler with Warmup

        if True:
            warmup_epochs = 4
            torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=fit_config["learning_rate_gamma"],
                                               verbose=False)

            scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                        warmup_start_value=learning_rate * 10 ** (-5),
                                                        warmup_end_value=learning_rate,
                                                        warmup_duration=warmup_epochs)

            trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        handler = Timer(average=False)
        handler.attach(trainer)
        handler.attach(evaluator)

        # https://pytorch.org/ignite/metrics.html#
        # https://pytorch.org/ignite/contrib/metrics.html#module-ignite.contrib.metrics.regression
        metrics = {
            "Mean Error": MeanError(),
            "Maximum Absolute Error": MaximumAbsoluteError(),
            "Mean Absolute Error": MeanAbsoluteError(),
            "Mean Squared Error": MeanSquaredError(),
            "R2": R2Score(),
            "Loss": Loss(loss_func),
        }

        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        log_interval = 1

        self.total_loss = 0.0

        # Ignite Tutorial for Model Checkpoint and Early Stopping
        # https://www.kaggle.com/code/vfdev5/pytorch-and-ignite-on-fruits-360

        """
        Best Model Saver
        """

        attach_to_trainer = val_loader is None or True

        def score_function(engine):
            output = engine.state.output
            metrics = engine.state.metrics
            loss = metrics['Loss'] if "Loss" in metrics else output

            score = -loss  # - self.total_loss / len(train_loader)
            if attach_to_trainer:
                score = - self.total_loss / len(train_loader)

            print("Total Loss:", self.total_loss / len(train_loader))
            print("Score:", score)
            # Early stopping and model checkpoint focus on highest scores
            return score

        best_models_dir = fit_config["best_models_dir"]

        if os.path.exists(best_models_dir + "/"):
            shutil.rmtree(best_models_dir + "/")
        best_model_saver = ModelCheckpoint(best_models_dir,  # folder where to save the best model(s)
                                           filename_prefix=f"model_{self.model_settings_id}_{self.id}",
                                           # filename prefix -> {filename_prefix}_{name}_{step_number}_{score_name}={abs(score_function_result)}.pth
                                           score_name="val_loss",
                                           score_function=score_function,
                                           n_saved=3,
                                           atomic=True,
                                           # objects are saved to a temporary file and then moved to final destination, so that files are guaranteed to not be damaged
                                           # save_as_state_dict=True,  # Save object as state_dict (True by default)
                                           create_dir=True,
                                           require_empty=False)

        if attach_to_trainer:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, best_model_saver, {"best_model": self})
        else:
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_model_saver, {"best_model": self})



        """
        Logging
        """
        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            print(
                f"Epoch [{engine.state.epoch}], Iter [{engine.state.iteration}] - Loss: {engine.state.output:.4f} (lr = {optimizer.param_groups[0]['lr']})")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(engine):
            print(f"Epoch {engine.state.epoch} - Loss: {engine.state.output:.4f} (duration {handler.value()} s)")

        if val_loader is not None:
            @trainer.on(Events.EPOCH_COMPLETED)
            def log_validation_loss(engine):
                evaluator.run(val_loader)
                metrics = evaluator.state.metrics
                loss = metrics["Loss"]
                loss_info = f"Validation Loss: {loss:.4f}"
                metrics_info = "    Metrics: \n"
                for metric in metrics:
                    metrics_info += f"            {metric}={metrics[metric]}\n"
                epoch_info = f"Epoch {engine.state.epoch}"
                print(f"{epoch_info} - {loss_info}")
                print(metrics_info)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_average_epoch_loss(engine):
            avg_loss = self.total_loss / len(engine.state.dataloader)
            print(f"Validation Epoch {engine.state.epoch} - Average Loss: {avg_loss:.4f}")

        @trainer.on(Events.EPOCH_STARTED)
        def reset_total_loss(engine):
            self.total_loss = 0.0

        @trainer.on(Events.ITERATION_COMPLETED)
        def accumulate_batch_loss(engine):
            self.total_loss += engine.state.output

        self.reset()
        start = time.time()
        trainer.run(train_loader, max_epochs=fit_config["max_epochs"])
        end = time.time()
        duration = end - start
        print(
            f"Training took: {duration} s (this is an average of {duration / trainer.state.epoch} s per epoch and of {duration / trainer.state.iteration} s per iteration)")

        #######################
        # Load the best model #
        #######################

        param_file_path = get_best_params_path(dir=best_models_dir, identifier=self.model_settings_id)

        if param_file_path is not None:
            reg_state_dict = torch.load(param_file_path, map_location=torch.device('cpu'))
            self.load_state_dict(reg_state_dict)
            print(f"Loaded best Params ({param_file_path}")
        else:
            raise Exception("An error occured and params from training could not be loaded")

        self.is_trained = True
        if os.path.exists(best_models_dir + "/"):
            shutil.rmtree(best_models_dir + "/")


class AptamerPredictionModel(nn.Module):

    def __init__(self, config, id=-1):
        super(AptamerPredictionModel, self).__init__()
        self.config = config
        self.id = id
        self.tokenizer = my_setup.TOKENIZER

        self.model = JointModel(config=config, id=id)

    def __call__(self, inputs, train_mode=False, output_attentions=False, *args, **kwargs):
        model = self.model
        optimizer = model.optimizer
        device = model.device

        if train_mode:
            model.train(True)
        else:
            model.eval()
            
        if not optimizer is None:
            optimizer.zero_grad()

        inputs = inputs.to(device)
        model_outputs = model(inputs, output_attentions=output_attentions)

        return model_outputs

    def reset(self):
        self.model.reset()

    def fit(self, measurements=None, data_loaders=None, max_epochs=None, batch_size=None, split=None,
            update_config=None):
        fit_config = dict(self.config["training_regression"])
        if max_epochs is not None:
            fit_config["max_epochs"] = max_epochs
        if batch_size is not None:
            fit_config["batch_size"] = batch_size
        if split is not None or split == "default":
            fit_config["split"] = split
        if update_config is not None:
            fit_config.update(update_config)

        train_loader, val_loader, test_loader = None, None, None
        if data_loaders is None:
            data_loaders = self._prepare_data(measurements, fit_config=fit_config)

        train_loader = data_loaders[0]
        if len(data_loaders) >= 2:
            val_loader = data_loaders[1]
        if len(data_loaders) >= 3:
            test_loader = data_loaders[2]

        # self.model.reset() # This reset is not required, as the model itself in _train_model before starting the training.
        self.model._train_model(fit_config=fit_config, train_loader=train_loader, val_loader=val_loader)
        return self.model

    def _prepare_data(self, measurements, fit_config):
        """
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    Transform data to suitable format
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       """

        tokenizer = self.tokenizer
        split = fit_config["split"]
        batch_size = fit_config["batch_size"]

        """
        Define train_loader and val_loader (val_loader is optionally None)
        """
        pin_memory = True if fit_config["device"] == "cpu" else False
        val_loader = None
        if split is None or len(split) == 0:
            train_dataset = SequenceToExpressionDataset(measurements=measurements.data, tokenizer=tokenizer)
        else:
            # ToDo: In case validation is implemented: Add to Measurements class, that only measured constructs are considered for validation and not artifically added ones
            datas = measurements.split(split=split, split_by_construct=True, shuffle=True)
            train_dataset, val_dataset = [SequenceToExpressionDataset(data, tokenizer=tokenizer) for data in datas]
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    shuffle=True, pin_memory=pin_memory, num_workers=0)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=pin_memory, num_workers=0)

        return train_loader, val_loader


class AptamerPredictionEnsembleModel(nn.Module):
    def __init__(self, config):
        super(AptamerPredictionEnsembleModel, self).__init__()
        self.config = config
        self.ensemble_size = config["ensemble"]["ensemble_size"]

        self.models = nn.ModuleList([AptamerPredictionModel(config=config, id=iE) for iE in range(self.ensemble_size)])

        self.num_processes = os.cpu_count() - (10 if os.cpu_count() > 30 else (4 if os.cpu_count() > 20 else 1))

    def __call__(self, inputs, train_mode=False, combine_outputs=True, evaluate_parallel=True, output_attentions=False, *args, **kwargs):
        model_outputs = [model(inputs=inputs, train_mode=train_mode, output_attentions=output_attentions, *args, **kwargs) for model in self.models]
        if output_attentions:
            attentions = [elem[1] for elem in model_outputs]
            model_outputs = [elem[0] for elem in model_outputs]
            attentions = [weights.unsqueeze(-1) for weights in attentions]
            attentions = torch.cat(attentions, dim=-1)

        model_outputs = [output.unsqueeze(-1) for output in model_outputs]
        model_outputs = torch.cat(model_outputs, dim=-1)

        if combine_outputs:
            model_outputs = torch.mean(model_outputs, dim=-1)

        if output_attentions:
            return model_outputs, attentions
        return model_outputs

    def reset(self):
        for model in self.models:
            model.reset()

    def _train_model(self, id, *args, **kwargs):
        print(f"Launched Training of Model {id}")
        self.models[id].fit(**args[0])
        print(f"Completed Training of Model {id}")
        return self.models[id].to("cpu")

    def fit(self, data_loaders=None, measurements=None, max_epochs=None, batch_size=None, split=None,
            train_parallel=True, *args, **kwargs):
        use_processes = False
        start = time.time()
        if train_parallel:

            best_models_dir = self.config["training_regression"]["best_models_dir"]

            num_processes = self.num_processes
            # num_processes = 50

            # Specify the hyperparameters for each model version
            arguments = [
                {"data_loaders": data_loaders, "measurements": measurements, "max_epochs": max_epochs,
                 "batch_size": batch_size, "split": split,
                 "update_config": {"best_models_dir": best_models_dir + f"/model_id={iM}"}
                 } for iM in range(self.ensemble_size)
            ]

            # This Reainitialization is necessary to allow multiprocess cuda training as the sharing of CUDA tensors among processes is not supported
            self.models = nn.ModuleList([AptamerPredictionModel(config=self.config)
                                         for iE in range(self.ensemble_size)])
            print("Params before Training:", list(self.models[0].model.parameters())[-1])

            print(f"Launching Thread Pool of size {num_processes}.")
            if use_processes:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # Use the pool to distribute the model training tasks
                    results = pool.starmap(self._train_model, enumerate(arguments))
                    pool.close()
                    pool.join()
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
                    # Use the executor to distribute the model training tasks
                    futures = [executor.submit(self._train_model, version, params)
                               for version, params in enumerate(arguments)]

                    # concurrent.futures.wait(futures)
                    executor.shutdown(wait=True)
                    results = [future.result() for future in futures]

            self.models = nn.ModuleList(results)
            print("Params after Training:", list(self.models[0].model.parameters())[-1])


        else:
            for model in self.models:
                model.fit(data_loaders=data_loaders, measurements=measurements, max_epochs=max_epochs,
                          batch_size=batch_size, split=split)

        end = time.time()
        print(
            f"Training of {self.ensemble_size} models took {end - start} s ({(end - start) / self.ensemble_size} s per model on average)")
        return self


if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # aptamer_model = AptamerPredictionModel(config)
    aptamer_model = AptamerPredictionEnsembleModel(config)

    constructs_file_path = config["data"]["constructs_path"]
    measurements_file_path = config["data"]["measurements_path"]

    measurements = Measurements(measurements_file_path=measurements_file_path,
                                constructs_file_path=constructs_file_path)

    aptamer_model.fit(measurements=measurements, split=None, max_epochs=config["training_regression"]["max_epochs"])

    file_path = config["data"]["candidates_path"]
    sequences = Sequences(file_path, limit=None)
    print("Loaded Candidate Sequences")

    data_dict = {"Sequence": sequences.data}
    df = pd.DataFrame(data=data_dict)
    df["Score"] = np.nan
    for combi in my_setup.LIGAND_COMBINATIONS:
        df[combi] = np.nan

    dataset = SequenceToExpressionDataset(measurements=df, tokenizer=my_setup.TOKENIZER)

    encoded_sequences = torch.tensor(np.array([np.array(entry[0]) for entry in dataset.data]))

    predictions = aptamer_model(encoded_sequences)
    predictions = np.array(predictions.detach())
    for iC, combi in enumerate(my_setup.LIGAND_COMBINATIONS):
        df[combi] = predictions[:, iC]

    rel_predictions = df[df["Sequence"].map(lambda seq: seq in measurements.data["Sequence"].values)]

    plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(2), np.arange(2), "--")
    colors = ["Green", "Blue", "Orange", "Red"]
    for measurement in measurements.data.iloc:
        cur_prediction = None
        for prediction in rel_predictions.iloc:
            if measurement["Sequence"] == prediction["Sequence"]:
                cur_prediction = prediction
                break

        for iC, combi in enumerate(my_setup.LIGAND_COMBINATIONS):
            ax.scatter(measurement[combi], cur_prediction[combi], c=colors[iC], label=combi)
        pass
    ax.set_xlabel("Measurement")
    ax.set_ylabel("Prediction")
    ax.legend()
    plt.show()
    pass
