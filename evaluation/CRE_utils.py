import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
import tqdm
from lightning import Trainer

from boda2 import boda
from boda2.boda.common import utils
from boda2.boda.common.utils import unpack_artifact, model_fn


# import boda

def get_time_stamp():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


def save_results(result_data, identifier):
    formatted_time = get_time_stamp()

    output_directory = f"_output/{identifier}_{len(result_data)}_runs_{formatted_time}/"
    os.makedirs(output_directory, exist_ok=True)

    for iD, df in enumerate(result_data):
        file_path = os.path.join(output_directory, f"results_model_{iD:03d}.tsv")
        df.to_csv(file_path, sep="\t")
    print(f"Saved outputs to directory {output_directory}")


def load_results(dir_path):
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
    file_paths = sorted(file_paths)
    results_data = []
    identifier = os.path.dirname(dir_path).split(os.sep)[-1].split("_")[0]
    for path in file_paths:
        df = pd.read_csv(path, sep="\t")
        results_data.append(df)

    return results_data, identifier


def split_data(data, split=[0.7, 0.3, 0.0]):
    train_size = int(len(data) * split[0])
    test_size = int(len(data) * split[2])
    val_size = len(data) - train_size - test_size

    indexes = np.random.permutation(np.arange(len(data)))
    train_indexes = indexes[:train_size]
    val_indexes = indexes[train_size:train_size + val_size]
    test_indexes = indexes[train_size + val_size:]

    # train_data = data[train_indexes]
    # val_data = data[val_indexes]
    # test_data = data[test_indexes]

    return train_indexes, val_indexes, test_indexes


def create_command(datafile_path, default_root_dir, artifact_path, min_epochs=60, max_epochs=200):
    command = ["python", "../boda2/src/train.py",
               "--data_module=MPRA_DataModule",
               f"--datafile_path={datafile_path}",  # Needs modification for ensemble (if bagging is done)
               "--sep tab --sequence_column sequence",
               "--activity_columns K562_log2FC HepG2_log2FC SKNSH_log2FC",
               "--stderr_columns K562_lfcSE HepG2_lfcSE SKNSH_lfcSE",
               "--stderr_threshold 1.0 --batch_size=1076",
               "--duplication_cutoff=0.5 --std_multiple_cut=6.0",
               "--val_chrs V --test_chrs E",
               "--synth_val_pct=0.0 --synth_test_pct=99.98",
               "--padded_seq_len=600 --use_reverse_complements=True --num_workers=8",
               "--model_module=BassetBranched",
               "--input_len 600",
               "--conv1_channels=300 --conv1_kernel_size=19",
               "--conv2_channels=200 --conv2_kernel_size=11",
               "--conv3_channels=200 --conv3_kernel_size=7",
               "--linear_activation=ReLU --linear_channels=1000",
               "--linear_dropout_p=0.11625456877954289",
               "--branched_activation=ReLU --branched_channels=140",
               "--branched_dropout_p=0.5757068086404574",
               "--n_outputs=3 --n_linear_layers=1",
               "--n_branched_layers=3 --n_branched_layers=3",
               "--use_batch_norm=True --use_weight_norm=False",
               "--loss_criterion=L1KLmixed --beta=5.0",
               "--reduction=mean",
               "--graph_module=CNNTransferLearning",
               "--parent_weights=../_data_evaluation/CRE/my-model.epoch_5-step_19885.pkl",  # Check effect
               "--frozen_epochs=0",
               "--optimizer=Adam --amsgrad=True",
               "--lr=0.0032658700881052086 --eps=1e-08 --weight_decay=0.0003438210249762151",
               "--beta1=0.8661062881299633 --beta2=0.879223105336538",
               "--scheduler=CosineAnnealingWarmRestarts --scheduler_interval=step",
               "--T_0=4096 --T_mult=1 --eta_min=0.0 --last_epoch=-1",
               "--checkpoint_monitor=entropy_spearman --stopping_mode=max",
               f"--stopping_patience=30 --accelerator=gpu --devices=1 --min_epochs={min_epochs} --max_epochs={max_epochs}",
               "--precision=16",
               f"--default_root_dir={default_root_dir}",  # Needs modification for ensemble
               f"--artifact_path={artifact_path}"  # Needs modification for ensemble
               ]
    return command


def train_model(train_data, val_data, test_data, output_folder, min_epochs=60, max_epochs=200, identifier=None,
                model_id=0):
    if identifier is None:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        identifier = formatted_time

    # Create joint datafile with unique identifier
    # Create unique subdirectory for default_root_dir
    # Create unique subdirectory for artifacts_path ?

    train_data["chr"] = "T"
    val_data["chr"] = "V"
    test_data["chr"] = "E"
    new_data = pd.concat((train_data, val_data, test_data))
    datafile_path = os.path.join(output_folder, f"custom_data_{identifier}.tsv")
    new_data.to_csv(datafile_path, sep="\t")

    # print("DataFile Path:", datafile_path)

    # default_root_dir = "/tmp/output/artifacts"                       # The directory where the lightning logs (including model checkpoints) are stored and the directory in which the artifact is created prior to compression
    # artifact_path = "_results/mpra_model/"                           # The output directory of the artifact containing the best model
    default_root_dir = os.path.join("tmp/output/artifacts/", f"model_{identifier}")
    artifact_path = os.path.join("_results/mpra_model/", f"model_{identifier}")

    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)
    os.makedirs(artifact_path)

    if os.path.exists(default_root_dir):
        shutil.rmtree(default_root_dir)
    os.makedirs(default_root_dir)

    device_id = model_id % torch.cuda.device_count()

    command = create_command(datafile_path, default_root_dir, artifact_path, min_epochs=min_epochs,
                             max_epochs=max_epochs)
    joined_command = " ".join(command)

    result = subprocess.run(f'bash -c "source ../../bashrc; source activate base; conda activate NANDRiboswitchDesignByBayesOpt; conda info; python3 -V; CUDA_VISIBLE_DEVICES={device_id} {joined_command}"', capture_output=True, shell=True)
    # print("STDOUT:", result.stdout)
    # print("")
    # print("STDERR:", result.stderr)
    artifact_files = os.listdir(artifact_path)

    if len(artifact_files) == 0:
        raise Exception(f"Training of model {identifier} was not successful")
    elif len(artifact_files) > 1:
        raise Exception(f"There are {len(artifact_files)} artifacts instead of 1")

    artifact_file_path = os.path.join(artifact_path, artifact_files[0])
    return artifact_file_path

def starmap_pool(func, iterable, wait_time = 1, sleep_time=1, pool_size=10, pool_name=""):
    results = []

    prefix = f"{pool_name}_" if pool_name else ""
    with ThreadPool(processes=pool_size) as pool:
        for elem in iterable:
            result = pool.apply_async(func, elem)
            results.append(result)
            time.sleep(wait_time)

        all_completed = False
        iX = 0
        while not all_completed:
            all_completed = True
            completed_results = []
            errornous_results = []
            for iR, result in enumerate(results):
                is_ready = result.ready()
                all_completed = all_completed and is_ready
                if is_ready:
                    completed_results.append(result)
                    if not result.successful():
                        errornous_results.append(result.get())

            print(
                f"{prefix}STATUS {iX}: {len(completed_results)} completed and {len(errornous_results)} failed ({len(results) - len(completed_results)} still running)")
            if not all_completed:
                time.sleep(sleep_time)
            iX += 1

        outputs = [result.get() for result in results]
        pool.close()
        pool.join()

    return outputs

"""
The following code follows the train tutorial of 
Copyright (c) 2025 Sagar Gosai, Rodrigo Castro
"""


def create_flank_builder(input_len):

    left_pad_len = (input_len - 200) // 2
    right_pad_len = (input_len - 200) - left_pad_len

    left_flank = boda.common.utils.dna2tensor(
        boda.common.constants.MPRA_UPSTREAM[-left_pad_len:]
    ).unsqueeze(0)
    # print(f'left flank shape: {left_flank.shape}')

    right_flank = boda.common.utils.dna2tensor(
        boda.common.constants.MPRA_DOWNSTREAM[:right_pad_len]
    ).unsqueeze(0)
    right_flank.shape
    # print(f'right flank shape: {right_flank.shape}')

    flank_builder = boda.common.utils.FlankBuilder(
        left_flank=left_flank,
        right_flank=right_flank,
    )

    flank_builder.cuda()
    return flank_builder


def prepare_model(model_path, model_dir, model_id=0):
    """
    Must be used single threaded only as otherwise conflicts with overlapping use of the directory /artifacts can occur.

    model_path: the path to the model
    returns: the model
    """
    # print("PREPARE MODEL MODEL_PATH:", model_path)
    # print("PREPARE MODEL MODEL_DIR:", model_dir)
    my_model, model_dir = load_model(model_path, download_path=model_dir, model_id=model_id)
    input_len = torch.load(os.path.join(model_dir, 'torch_checkpoint.pt'), weights_only=False)[
        'model_hparams'].input_len
    flank_builder = create_flank_builder(input_len)
    print(f"Loaded model from {model_path}")
    return my_model, flank_builder


def evaluate_model(sequences, sequence_loader, model, flank_builder, model_id=0):
    results = []

    device = torch.device(f'cuda:{model_id % torch.cuda.device_count()}')
    flank_builder.to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(sequence_loader)):
            prepped_seq = flank_builder(batch[0].to(device))
            predictions = model(prepped_seq) + \
                          model(prepped_seq.flip(dims=[1, 2]))  # Also
            predictions = predictions.div(2.)
            results.append(predictions.detach().cpu())

    predictions = torch.cat(results, dim=0)

    pred_df = pd.DataFrame(predictions.numpy(), columns=['K562_preds', 'HepG2_preds', 'SKNSH_preds'])
    seq_df = pd.DataFrame(sequences, columns=["sequence"])
    pred_df = pd.concat((seq_df, pred_df), axis=1)
    return pred_df


"""
The following code is an excerpt from https://github.com/sjgosai/boda2/blob/main/boda/common/utils.py#L572 
to fix the weights_only=True default load issue arising from PyTorch 2.6 on.
Further modifications affecting file handling are made to allow parallel usage of the models.
"""
"""
MIT License

Copyright (c) 2025 Sagar Gosai, Rodrigo Castro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tarfile
from boda2.boda import model as _model


def unpack_artifact(artifact_path, download_path='./'):
    """
    Unpack a tar archive artifact.

    Args:
        artifact_path (str): Path to the artifact.
        download_path (str, optional): Path to extract the artifact. Defaults to './'.
    """
    print("Artifact Path:", artifact_path)
    if 'gs' in artifact_path:
        subprocess.call(['gsutil', 'cp', artifact_path, download_path])
        if os.path.isdir(download_path):
            tar_model = os.path.join(download_path, os.path.basename(artifact_path))
        elif os.path.isfile(download_path):
            tar_model = download_path
    else:
        assert os.path.isfile(artifact_path), "Could not find file at expected path."
        tar_model = artifact_path

    assert tarfile.is_tarfile(tar_model), f"Expected a tarfile at {tar_model}. Not found."

    shutil.unpack_archive(tar_model, download_path)
    print(f'archive unpacked in {download_path}', file=sys.stderr)


def model_fn(model_dir):
    """
    Load a model from a directory.

    Args:
        model_dir (str): Path to the model directory.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    # print("Loading Torch Weights:", os.path.abspath(os.path.join(model_dir,'torch_checkpoint.pt')))
    checkpoint = torch.load(os.path.join(model_dir, 'torch_checkpoint.pt'), weights_only=False)
    # print("Loaded Torch Weights:", os.path.abspath(os.path.join(model_dir,'torch_checkpoint.pt')))
    model_module = getattr(_model, checkpoint['model_module'])
    # print("Attributes")
    # TEST = vars(checkpoint['model_hparams'])
    # print("VARS")
    model = model_module(**vars(checkpoint['model_hparams']))
    # print("Module")
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    return model


def load_model(artifact_path, download_path="./", model_id=0):
    USE_CUDA = torch.cuda.device_count() >= 1

    if os.path.isdir(download_path):
        shutil.rmtree(download_path)

    unpack_artifact(artifact_path, download_path)

    model_dir = os.path.join(download_path, "artifacts/")

    my_model = model_fn(model_dir)
    my_model.eval()
    if USE_CUDA:
        device = torch.device(f'cuda:{model_id % torch.cuda.device_count()}')
        # my_model.cuda()
        my_model.to(device)

    return my_model, model_dir


"""
The following code is an excerpt from https://github.com/sjgosai/boda2/blob/main/boda/common/utils.py 
"""
"""
MIT License

Copyright (c) 2025 Sagar Gosai, Rodrigo Castro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def dna2tensor(sequence_str, vocab_list=["A", "C", "G", "T"]):
    """
    Convert a DNA sequence to a one-hot encoded tensor.

    Args:
        sequence_str (str): DNA sequence string.
        vocab_list (list): List of DNA nucleotide characters.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the sequence.
    """
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letterIdx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letterIdx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor


"""
Modified from https://github.com/sjgosai/boda2/blob/main/src/train.py
"""


def arg_parser(custom_args):
    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True, help='BODA data module to process dataset.')
    group.add_argument('--model_module', type=str, required=True, help='BODA model module to fit dataset.')
    group.add_argument('--graph_module', type=str, required=True, help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/',
                       help='Path where model artifacts are deposited.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    group.add_argument('--checkpoint_monitor', type=str, help='String to monior PTL logs if saving best.')
    group.add_argument('--stopping_mode', type=str, default='min', help='Goal for monitored metric e.g. (max or min).')
    group.add_argument('--stopping_patience', type=int, default=100,
                       help='Number of epochs of non-improvement tolerated before early stopping.')
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False,
                       help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')
    known_args, leftover_args = parser.parse_known_args(custom_args[1:])

    Data = getattr(boda.data, known_args.data_module)
    Model = getattr(boda.model, known_args.model_module)
    Graph = getattr(boda.graph, known_args.graph_module)

    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)

    known_args, leftover_args = parser.parse_known_args(custom_args[1:])

    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)

    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')

    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args(custom_args[1:])
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args(custom_args[1:])

    args = boda.common.utils.organize_args(parser, args)
    return args
