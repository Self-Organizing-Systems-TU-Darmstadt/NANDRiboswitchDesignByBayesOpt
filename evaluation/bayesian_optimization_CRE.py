import json
import os
import shutil
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from datetime import datetime

import tqdm
import yaml
from torch import nn

import my_setup
from bayesian_optimization.acquisition_functions import ThompsonSampling, UpperConfidenceBound
from boda2.analysis.SG011__GPytorch.single_gpu_gp import output
from data_processing.masked_sequence_model_token_based_triplet_dataset import Sequences
from data_processing.sequence_to_expression_dataset import Measurements, SequenceToExpressionDataset
# from evaluation.CRE_utils import *
from evaluation.CRE_utils import dna2tensor, train_model, split_data, get_time_stamp, prepare_model, evaluate_model
from evaluation.Model_Trainer_and_Executer import identifier
from models.aptamer_prediction_model import AptamerPredictionEnsembleModel

from multiprocessing import Pool


class PromoterEnsembleModel:
    def __init__(self, config):
        self.config = config
        self.ensemble_size = config["ensemble"]["ensemble_size"]

        self.model_paths = [None] * self.ensemble_size

        self.pool_size = np.min([torch.cuda.device_count() * 2, os.cpu_count() - 6])

    def __call__(self, domain, train_mode=False, combine_outputs=True, evaluate_parallel=True, output_attentions=False,
                 *args, **kwargs):
        def apply_model(model_path, model_id, sequences, seq_loader):

            model, flank_builder = prepare_model(model_path, model_dir=os.path.join("./_intermediate/", identifier),
                                                 model_id=model_id)
            pred_df = evaluate_model(sequences=sequences, sequence_loader=seq_loader, model=model,
                                     flank_builder=flank_builder, model_id=model_id)
            return pred_df

        model_paths = self.model_paths
        if any(map(lambda x: x is None, model_paths)):
            raise Exception("Model is not trained!")


        argument_list = [(model_path, iX, domain[0], domain[1]) for iX, model_path in enumerate(self.model_paths)]

        with ThreadPool(self.pool_size) as pool:
            # model_path = train_model(*args)
            results = pool.starmap(apply_model, argument_list)

        model_outputs = None

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

    def fit(self, measurements=None, *args, **kwargs):
        split = [0.75, 0.25, 0.0]

        time_stamp = get_time_stamp()

        argument_list = []
        for iX in range(self.ensemble_size):
            # Bagging by random subsampling of the dataset
            train_indexes, validation_indexes, test_indexes = split_data(measurements, split=split)
            train_data = pd.DataFrame(measurements.iloc[train_indexes])
            val_data = pd.DataFrame(measurements.iloc[validation_indexes])
            test_data = pd.DataFrame(measurements.iloc[test_indexes])
            if split[2] == 0.0:
                test_data = pd.DataFrame(measurements.iloc[:2])

            identifier = f"model_{iX:03d}"
            output_folder = os.path.join("../_data_evaluation/CRE/custom/training_data/", f"{time_stamp}_{identifier}")
            os.makedirs(output_folder, exist_ok=True)
            argument_list.append((train_data, val_data, test_data, output_folder, 60, 200, identifier, iX))

        start = time.time()
        with Pool(self.pool_size) as pool:

            # model_path = train_model(*args)
            pool.starmap(train_model, argument_list)

        end = time.time()
        print(
            f"Training of {self.ensemble_size} models took {end - start} s ({(end - start) / self.ensemble_size} s per model on average)")

        # Delete Training data again
        for args in argument_list:
            shutil.rmtree(args[3], ignore_errors=True)
        return self


# def model_based_optimization_loop(data, n_init, n_rounds, q, identifier="SingleRun", model_id=0, output_directory=None):
#     split = [0.7, 0.3, 0.0]
#
#     time_stamp = get_time_stamp()
#
#     output_file = None
#     if output_directory:
#         # output_dir = f"_output_backup/"
#         output_dir = output_directory
#         output_file = os.path.join(output_dir, f"{identifier}_{time_stamp}.tsv")
#         os.makedirs(output_dir, exist_ok=True)
#
#     print(f"{identifier}: Preparing Sequences")
#     sequences = data["sequence"]
#     seq_tensor = torch.stack([dna2tensor(seq) for seq in tqdm.tqdm(sequences, total=len(sequences))],
#                              dim=0)
#     seq_dataset = torch.utils.data.TensorDataset(seq_tensor)
#     seq_loader = torch.utils.data.DataLoader(seq_dataset, batch_size=128)
#
#     print(f"{identifier}: Creating Initial Dataset")
#     initial_data_indexes = np.random.choice(np.arange(len(data)), replace=False, size=n_init)
#     initial_data = data.iloc[initial_data_indexes]
#
#     result_data = pd.DataFrame(initial_data)
#     # result_data_indexes = initial_data_indexes.copy()
#
#     result_data["round"] = 0
#
#     for iR in range(n_rounds):
#         train_indexes, validation_indexes, test_indexes = split_data(result_data, split=split)
#         train_data = pd.DataFrame(data.iloc[train_indexes])
#         val_data = pd.DataFrame(data.iloc[validation_indexes])
#         test_data = pd.DataFrame(data.iloc[test_indexes])
#         if split[2] == 0.0:
#             test_data = pd.DataFrame(data.iloc[:2])
#
#         candidate_sequences = data.loc[np.logical_not(data["sequence"].isin(list(result_data["sequence"])))]
#
#         print(f"{identifier} {iR}: Training Model")
#         model_path = train_model(train_data, val_data, test_data, min_epochs=6, max_epochs=20, identifier=identifier,
#                                  model_id=model_id)
#         print(f"{identifier} {iR}: Trained Model ({model_path})")
#         model, flank_builder = prepare_model(model_path, model_dir=os.path.join("./_intermediate/", identifier),
#                                              model_id=model_id)
#         print(f"{identifier} {iR}: Prepared Model")
#         pred_df = evaluate_model(sequences=sequences, sequence_loader=seq_loader, model=model,
#                                  flank_builder=flank_builder, model_id=model_id)
#         print(f"{identifier} {iR}: Evaluated Model")
#
#         pred_df["score"] = pred_df.apply(scoring_function, axis=1)
#         pred_df = pred_df.sort_values(by="score", ascending=False)
#
#         candidates = pred_df.iloc[:q]
#
#         new_data = pd.DataFrame(data.loc[candidates.index])
#         new_data["round"] = iR + 1
#         result_data = pd.concat((result_data, new_data))
#
#         if output_file:
#             result_data.to_csv(output_file, sep="\t")
#         print(f"{identifier} {iR}: Currently highest score {np.max(result_data['score'])}")
#
#     return result_data


def main(measurements, candidate_sequences, identifier=""):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    multiprocessing.set_start_method("spawn")

    my_setup.DEVICE = "cpu"
    my_setup.DEVICE_TRAINING = "cpu"
    """
    Setup
    """

    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    suffix = "" if identifier == "" else f"_{identifier}"
    results_dir = os.path.join("_results", formatted_timestamp + suffix + os.sep)
    os.makedirs(results_dir)

    config["log_dir"] = results_dir
    bo_config = config["bayesian_optimization"]
    acquisition_batch_size = bo_config["acquisition_batch_size"]
    ensemble_size = config["ensemble"]["ensemble_size"]

    # acquisition_method = ThompsonSampling
    acquisition_method = UpperConfidenceBound

    """
    Load the data
    """
    # Training Data
    measurements = measurements
    training_data_size = len(measurements)

    # Domain Data
    seq_tensor = torch.stack(
        [dna2tensor(seq) for seq in tqdm.tqdm(candidate_sequences, total=len(candidate_sequences))],
        dim=0)
    seq_dataset = torch.utils.data.TensorDataset(seq_tensor)
    seq_loader = torch.utils.data.DataLoader(seq_dataset, batch_size=128)

    domain = (candidate_sequences, seq_loader)

    """
    Setup the model
    """
    model = PromoterEnsembleModel(config)
    print("Model Created")

    model_output_transform = None
    if not bool(config["training_regression"]["predict_score"]):
        def model_output_transform(model_outputs):
            # Transform Model Outputs to Score
            # Model_outputs is of shape M x 3 x N with M the number of sequences and N as the ensemble size.
            # The desired value is of shape M x 1 x N
            M, nV, N = model_outputs.shape
            transformed_output = np.empty(shape=(M, 1, N))
            for iM in range(M):
                for iN in range(N):
                    expression_levels = {combi: model_outputs[iM, iC, iN] for iC, combi in
                                         enumerate(my_setup.LIGAND_COMBINATIONS)}
                    score = my_setup.SCORE_FUNC(expression_levels)
                    transformed_output[iM, 0, iN] = score
            raise Exception("Check Validit")
            return transformed_output

    """
    Perform Batch Acquisition Step
    """
    fit_args = {}

    start_time = time.time()
    ac_func = acquisition_method(model=model, domain=domain, data=measurements, model_args=fit_args,
                                 model_output_transform=model_output_transform, config=config)
    proposals = ac_func(acquisition_batch_size)

    end_time = time.time()
    duration = end_time - start_time

    """
    Output results
    """
    print("\n\n--------------------------------------------------------")
    print(
        f"The single Bayesian Optimization step took {duration} seconds ({duration * 1.0 / acquisition_batch_size} s per proposal).")
    print("The sequences selected for further evaluation are: ")
    for elem in proposals:
        print(elem, ":", proposals[elem])
    print("--------------------------------------------------------")

    """
    Store the results
    """

    predictions_results_path = f"{results_dir}/bayesian_optimization_run_results_{formatted_timestamp}.xlsx"
    proposals_path = f"{results_dir}/bayesian_optimization_proposals_{formatted_timestamp}.json"
    measurements.data.to_excel(predictions_results_path)

    results_info = {"Method": acquisition_method.__name__,
                    "Ensemble Size": ensemble_size,
                    "Acquisition Batch Size": acquisition_batch_size,
                    "Execution Time": duration,
                    "Timestamp": formatted_timestamp,
                    "Domain Size": len(domain[0]),
                    "Training Data Size": training_data_size}
    results_info["Proposals"] = proposals
    with open(proposals_path, "w") as file:
        json.dump(results_info, file, indent=4)

    pass


"""
Refactor to original version and instead add an additional script that calls this code and simulates the experimental loop
"""

"""
This script performs multiple iterations of batch Bayesian optimization utilizing a BatchAcquisitionFunction
The evaluation is done in silico with the pre-existing dataset: _data_evaluation/CRE/41586_2024_8070_MOESM4_ESM_Malinois_Training_Data.txt 
"""
if __name__ == '__main__':
    measurements = pd.read_csv("../_data_evaluation/CRE/custom/measurements.tsv", sep="\t")
    with open("../_data_evaluation/CRE/custom/candidate_sequences.txt") as file:
        candidate_sequences = file.read().splitlines()

    main(measurements, candidate_sequences)
