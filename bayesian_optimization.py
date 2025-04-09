import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from datetime import datetime

import yaml

import my_setup
from bayesian_optimization.acquisition_functions import ThompsonSampling, UpperConfidenceBound
from data_processing.masked_sequence_model_token_based_triplet_dataset import Sequences
from data_processing.sequence_to_expression_dataset import Measurements, SequenceToExpressionDataset
from models.aptamer_prediction_model import AptamerPredictionEnsembleModel

"""
This script performs one iteration of batch bayesian optimization utilizing a BatchAcquisitionFunction
"""
if __name__ == '__main__':
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

    results_dir = "_results/"
    results_dir += f"{formatted_timestamp}/"
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
    # Training Data #
    constructs_file_path = config["data"]["constructs_path"]
    measurements_file_path = config["data"]["measurements_path"]
    candidates_path = config["data"]["candidates_path"]

    measurements = Measurements(measurements_file_path=measurements_file_path,
                                constructs_file_path=constructs_file_path)

    training_data_size = len(measurements.data)

    # Domain Data

    sequences = Sequences(candidates_path, limit=None)
    print("Loaded Sequences")

    data_dict = {"Sequence": sequences.data}
    df = pd.DataFrame(data=data_dict)
    df["Score"] = np.nan
    for combi in my_setup.LIGAND_COMBINATIONS:
        df[combi] = np.nan

    dataset = SequenceToExpressionDataset(measurements=df, tokenizer=my_setup.TOKENIZER)

    encoded_sequences = torch.tensor(np.array([np.array(entry[0]) for entry in dataset.data]))

    domain = (sequences.data, encoded_sequences)

    """
    Setup the model
    """
    model = AptamerPredictionEnsembleModel(config)
    print("Model Created")

    model_output_transform = None
    if not bool(config["training_regression"]["predict_score"]):
        def model_output_transform(model_outputs):
            # Transform Model Outputs to Score
            # Model_outputs is of shape M x 4 x N with M the number of sequences and N as the ensemble size.
            # The desired value is of shape M x 1 x N
            M, nV, N = model_outputs.shape
            transformed_output = np.empty(shape=(M, 1, N))
            for iM in range(M):
                for iN in range(N):
                    expression_levels = {combi: model_outputs[iM, iC, iN] for iC, combi in
                                         enumerate(my_setup.LIGAND_COMBINATIONS)}
                    score = my_setup.SCORE_FUNC(expression_levels)
                    transformed_output[iM, 0, iN] = score

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
