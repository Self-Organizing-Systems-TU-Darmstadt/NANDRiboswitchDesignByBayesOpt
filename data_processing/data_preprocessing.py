"""
@author: Erik Kubaczka

"""

import torch
from torch.utils.data import DataLoader

import my_setup
from data_processing.masked_sequence_model_token_based_triplet_dataset import Sequences, SequenceToSequenceDataset



def split_data(dataset, split=None):
    if split is None:
        split = [0.7, 0.15, 0.15]
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, split)
    return train_set, val_set, test_set


def create_datasets(file_path, n_repetitions=1, split=None, limit=None, tokenizer=None):
    sequences = Sequences(file_path, limit=limit)

    sequences_split = sequences.split(split=split)
    datasets = [SequenceToSequenceDataset(data, n_repetitions=n_repetitions, tokenizer=tokenizer)
                for data in sequences_split]

    return datasets


def create_data_loaders(file_path, batch_size=32, n_repetitions=-1, split=None, limit=None, tokenizer=None):
    datasets = create_datasets(file_path=file_path, split=split, n_repetitions=n_repetitions, limit=limit, tokenizer=tokenizer)
    # We need to use "pin_memory = False" as the tensors are already transferred to the computing device.
    pin_memory = True if my_setup.DEVICE == "cpu" else False
    data_loaders = [
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0)
        for dataset in datasets]
    return data_loaders