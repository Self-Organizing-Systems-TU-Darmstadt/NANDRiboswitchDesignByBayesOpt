import json
import os
import shutil
import time
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
import tqdm
import yaml
from scipy.stats import norm

import my_setup
from bayesian_optimization.acquisition_functions import UpperConfidenceBound, BatchAcquisitionFunction
# from evaluation.CRE_utils import *
from evaluation.CRE_utils import dna2tensor, train_model, split_data, get_time_stamp, prepare_model, evaluate_model


class Measurements:

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)


class PromoterEnsembleModel:
    def __init__(self, config):
        self.config = config
        self.ensemble_size = config["ensemble"]["ensemble_size"]
        self.ensemble_size = 2

        self.model_paths = [None] * self.ensemble_size

        self.pool_size = np.min([torch.cuda.device_count() * 2, os.cpu_count() - 6])

    def __call__(self, domain, train_mode=False, combine_outputs=True, evaluate_parallel=True, output_attentions=False,
                 *args, **kwargs):
        def apply_model(model_path, identifier, model_id, sequences, seq_loader):

            model, flank_builder = prepare_model(model_path, model_dir=os.path.join("./_intermediate/", identifier),
                                                 model_id=model_id)
            pred_df = evaluate_model(sequences=sequences, sequence_loader=seq_loader, model=model,
                                     flank_builder=flank_builder, model_id=model_id)
            return pred_df

        self.model_paths = ["_results/mpra_model/model_model_001\model_artifacts__20251216_162628__688811.tar.gz",
                            "_results/mpra_model/model_model_000\model_artifacts__20251216_162627__634186.tar.gz"]
        model_paths = self.model_paths
        if any(map(lambda x: x is None, model_paths)):
            raise Exception("Model is not trained!")

        argument_list = [(model_path, f"model_{iX:03d}", iX, domain[0], domain[1]) for iX, model_path in
                         enumerate(self.model_paths)]

        with ThreadPool(self.pool_size) as pool:
            # model_path = train_model(*args)
            results = pool.starmap(apply_model, argument_list)

        model_outputs = np.stack([elem[["K562_preds", "HepG2_preds", "SKNSH_preds"]] for elem in results], axis=-1)

        if combine_outputs:
            model_outputs = torch.mean(model_outputs, dim=-1)

        return model_outputs

    def fit(self, measurements=None, *args, **kwargs):
        return
        split = [0.75, 0.25, 0.0]

        data = measurements.data

        time_stamp = get_time_stamp()

        argument_list = []
        for iX in range(self.ensemble_size):
            # Bagging by random subsampling of the dataset
            train_indexes, validation_indexes, test_indexes = split_data(data, split=split)
            train_data = pd.DataFrame(data.iloc[train_indexes])
            val_data = pd.DataFrame(data.iloc[validation_indexes])
            test_data = pd.DataFrame(data.iloc[test_indexes])
            if split[2] == 0.0:
                test_data = pd.DataFrame(data.iloc[:2])

            identifier = f"model_{iX:03d}"
            output_folder = os.path.join("../_data_evaluation/CRE/custom/training_data/", f"{time_stamp}_{identifier}")
            os.makedirs(output_folder, exist_ok=True)
            argument_list.append((train_data, val_data, test_data, output_folder, 1, 1, identifier, iX))

        start = time.time()
        with ThreadPool(self.pool_size) as pool:
            # model_path = train_model(*args)
            model_paths = pool.starmap(train_model, argument_list)
        end = time.time()

        self.model_paths = model_paths
        print(
            f"Training of {self.ensemble_size} models took {end - start} s ({(end - start) / self.ensemble_size} s per model on average)")

        # Delete Training data again
        for args in argument_list:
            shutil.rmtree(args[3], ignore_errors=True)
        return self


class UpperConfidenceBound(BatchAcquisitionFunction):
    def __init__(self, model, model_args, domain, data, model_output_transform=None, config=None):
        super().__init__(model, model_args, domain, data, model_output_transform=model_output_transform, config=config)
        self.label = "UpperConfidenceBound"
        self.coverage_probability = config["bayesian_optimization"]["coverage_probability"]
        self.normal_approximation_mode = config["bayesian_optimization"]["normal_approximation_mode"]
        self.beta = norm().ppf(self.coverage_probability)
        self.std_dev_scaler = 1
        pass

    def utility_function(self):
        domain = self.domain
        start = time.time()
        model_outputs = self.model(domain, combine_outputs=False)

        end = time.time()
        duration = end - start
        print(
            f"Model Evaluation for {len(domain[0])} entries took {duration} s ({duration / len(domain[0])} s per sample)",
            file=self.log_file)

        transformed_output = model_outputs
        # In case the output is the expression level, transform to the score first and derive then the UCB
        if self.model_output_transform is not None:
            transformed_output = self.model_output_transform(model_outputs)
        # transformed_output is of shape M x 1 x N with M the number of sequences and N the ensemble size

        means = np.mean(transformed_output, axis=-1)
        if self.normal_approximation_mode:
            # Determination of the UCB via a normal distribution approximation of the data

            std_devs = np.std(transformed_output, axis=-1)
            means = means[:, 0]
            std_devs = std_devs[:, 0]
            ucb = means + self.beta * self.std_dev_scaler * std_devs
        else:
            ensemble_size = transformed_output.shape[-1]
            index = ensemble_size * self.coverage_probability
            index = index - 1  # Shift to 0 to N-1 indexing
            if index - int(index) == 0:
                # In case the index is an integer value, take the average of the current and next
                index = int(index)
                indexes = [index, index + 1]
            else:
                indexes = [int(np.ceil(index))]  # Round up to the next full Integer

            sorted_output = np.sort(transformed_output, axis=-1)
            ucb = np.mean(sorted_output[:, 0, indexes], axis=-1, dtype=float)

        return means, ucb, model_outputs


def main(measurements, candidate_sequences, identifier=""):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # multiprocessing.set_start_method("spawn")

    my_setup.DEVICE = "cpu"
    my_setup.DEVICE_TRAINING = "cpu"
    conditions = ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"]
    my_setup.LIGAND_COMBINATIONS = conditions

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

    scoring_function = lambda elem: elem["K562_log2FC"] if "K562_log2FC" in elem else elem["K562_preds"]
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
                    expression_levels = {combi: model_outputs[iM, iC, iN]
                                         for iC, combi in enumerate(conditions)}
                    score = scoring_function(expression_levels)
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


"""
Refactor to original version and instead add an additional script that calls this code and simulates the experimental loop
"""

"""
This script performs multiple iterations of batch Bayesian optimization utilizing a BatchAcquisitionFunction
The evaluation is done in silico with the pre-existing dataset: _data_evaluation/CRE/41586_2024_8070_MOESM4_ESM_Malinois_Training_Data.txt 
"""
if __name__ == '__main__':
    measurements = pd.read_csv("../_data_evaluation/CRE/custom/measurements.tsv", sep="\t")
    measurements["Sequence"] = measurements["sequence"]
    measurements = Measurements(measurements)
    with open("../_data_evaluation/CRE/custom/candidate_sequences.txt") as file:
        candidate_sequences = file.read().splitlines()

    candidate_sequences = candidate_sequences[:1000]

    main(measurements, candidate_sequences)
