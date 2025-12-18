import os

import numpy as np
import pandas as pd
import yaml

from evaluation import bayesian_optimization_CRE
from evaluation.CRE_utils import get_time_stamp


def standalone_data_creation():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    bo_config = config["bayesian_optimization"]

    n_init = bo_config["n_init"]
    n_rounds = bo_config["n_rounds"]
    q = bo_config["acquisition_batch_size"]

    """
    Prepare Data
    """
    file_path = "../_data_evaluation/CRE/41586_2024_8070_MOESM4_ESM.txt"
    scoring_function = lambda elem: elem["K562_log2FC"] if "K562_log2FC" in elem.index else elem["K562_preds"]

    data = pd.read_csv(file_path, delimiter="\t", low_memory=False)
    data = data.loc[data.loc[:, ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0]
    data = data.loc[data['sequence'].str.len() == 200].reset_index(drop=True)
    data["score"] = data.apply(scoring_function, axis=1)

    data = data.loc[np.logical_not(np.isnan(data["score"]))]

    print(f"Maximum Score is {data['score'].max()}")

    """
    Each Bayesian Optimization run needs two datasets
    - Measurements Dataframe
    - Candidates (same for all) 
    """

    output_directory = os.path.join("../", "_data_evaluation", "CRE", "custom")
    os.makedirs(output_directory, exist_ok=True)

    # Candidates
    candidate_sequences = list(data["sequence"])
    candidates_file = os.path.join(output_directory, "candidate_sequences.txt")
    with open(candidates_file, "w") as file:
        file.write("\n".join(candidate_sequences))

    # Measurements
    initial_data_indexes = np.random.choice(np.arange(len(data)), replace=False, size=n_init)
    initial_data = data.iloc[initial_data_indexes]

    measurements = pd.DataFrame(initial_data)
    measurements_file = os.path.join(output_directory, "measurements.tsv")
    measurements.to_csv(measurements_file, index=False, sep="\t")


def integrated_bayesian_optimization():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    bo_config = config["bayesian_optimization"]

    n_init = bo_config["n_init"]
    n_rounds = bo_config["n_rounds"]
    q = bo_config["acquisition_batch_size"]

    """
    Prepare Data
    """
    file_path = "../_data_evaluation/CRE/41586_2024_8070_MOESM4_ESM.txt"
    scoring_function = lambda elem: elem["K562_log2FC"] if "K562_log2FC" in elem.index else elem["K562_preds"]

    data = pd.read_csv(file_path, delimiter="\t", low_memory=False)
    data = data.loc[data.loc[:, ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0]
    data = data.loc[data['sequence'].str.len() == 200].reset_index(drop=True)
    data["score"] = data.apply(scoring_function, axis=1)
    data["Sequence"] = data["sequence"]

    data = data.loc[np.logical_not(np.isnan(data["score"]))]
    data = pd.DataFrame(data.iloc[:1000])

    print(f"Maximum Score is {data['score'].max()}")

    """
    Each Bayesian Optimization run needs two datasets
    - Measurements Dataframe
    - Candidates (same for all) 
    """
    time_stamp = get_time_stamp()

    output_directory = os.path.join("_results", time_stamp)

    full_bo_run(data, n_init, n_rounds, identifier="bayesian_optimization_test", output_directory=output_directory)


def full_bo_run(data, n_init, n_rounds, identifier="SingleRun", output_directory=None):
    time_stamp = get_time_stamp()

    output_file = None
    if output_directory:
        output_dir = output_directory
        output_file = os.path.join(output_dir, f"{identifier}_{time_stamp}.tsv")
        output_file_bo = lambda iR: output_file.replace(".tsv", f"_BO_round_{iR}.tsv")
        os.makedirs(output_dir, exist_ok=True)

    initial_data_indexes = np.random.choice(np.arange(len(data)), replace=False, size=n_init)
    initial_data = data.iloc[initial_data_indexes]

    result_data = pd.DataFrame(initial_data)
    candidate_sequences = list(data["sequence"])

    for iR in range(n_rounds):
        measurements = pd.DataFrame(result_data)
        measurements["Type"] = "Measurement"
        measurements = bayesian_optimization_CRE.Measurements(measurements)
        proposals, results_info = bayesian_optimization_CRE.main(measurements, candidate_sequences,
                                                                 identifier=identifier)
        new_data = data.loc[data["sequence"].isin(list(proposals.keys()))]
        new_data = pd.DataFrame(new_data)
        new_data["round"] = iR + 1
        result_data = pd.concat((result_data, new_data))

        if output_file:
            result_data.to_csv(output_file, sep="\t", index=False)

        if output_file_bo:
            measurements.data.to_csv(output_file_bo(iR), sep="\t", index=False)

        print(f"{identifier} {iR}: Currently highest score {np.max(result_data['score'])}")


if __name__ == '__main__':
    # standalone_data_creation()
    integrated_bayesian_optimization()

    pass
