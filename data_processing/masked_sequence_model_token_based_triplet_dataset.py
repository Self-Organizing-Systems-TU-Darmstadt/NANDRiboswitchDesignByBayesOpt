import json
import os

import Levenshtein
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import my_setup
from data_processing import tokenization
from data_processing.sequence_utils import mask_tokens, prepare_sequence


class Sequences:

    def __init__(self, file_path=None, sequences=None, limit=None, shuffle=False):
        assert file_path is not None or sequences is not None
        data = sequences
        if file_path is not None:

            # Read the _data from the file and preprocess it
            with open(file_path, 'r') as file:
                seqs = file.readlines()

            seqs = map(lambda val: val.strip(), seqs)
            data = list(seqs)


        data = map(prepare_sequence, data)
        data = set(data)
        data = list(data)



        data = [seq for seq in data
                if seq is not None
                and len(set(seq).difference(my_setup.ALLOWED_BASES)) == 0
                and len(seq) <= my_setup.MAX_SEQUENCE_LENGTH]

        if shuffle:
            np.random.shuffle(data)

        if limit is not None and type(limit) is int and limit < len(data):
            data = data[:limit]

        self.data = data
        pass

    def repeat(self, n):
        self.data = self.data * n

    def split(self, split=None):
        if split is None:
            split = [0.7, 0.15, 0.15]

        training, test = train_test_split(self.data, train_size=split[0], shuffle=True)
        if len(split) == 3:
            validation, test = train_test_split(test, test_size=split[2] / (split[1] + split[2]))
            return training, validation, test

        return training, test


class SequenceToSequenceDataset(Dataset):
    def __init__(self, sequences, n_repetitions=None, max_sequence_length=-1, tokenizer=None, online_masking=True,
                 triplets=None):

        self.data = []
        if max_sequence_length < 0:
            max_sequence_length = my_setup.MAX_SEQUENCE_LENGTH
        self.max_input_length = max_sequence_length + 2  # The two additional token are the [REG] and the [SEP] token at the start and end respectively
        if n_repetitions is None:
            self.n_repetitions = 1
        else:
            self.n_repetitions = max(n_repetitions, 1)

        if tokenizer is None:
            tokenizer = my_setup.TOKENIZER

        if triplets is None:
            triplets = my_setup.TRIPLET_MODE

        classic_triplets = my_setup.TRIPLET_MODE_CLASSIC_NEGATIVE_SELECTION

        online_masking = True

        self.tokenizer = tokenizer
        self.online_masking = online_masking
        self.triplets = triplets
        self.classic_triplets = classic_triplets

        # 1. Tokenize
        # 2. Repeat Sequences
        # 3. Mask
        # 4. Padd
        # 5. Encode Tokens

        if online_masking:
            data = sequences
        else:
            # 1. Tokenize
            tokenized_sequences = map(tokenizer.tokenize, sequences)
            tokenized_sequences = list(tokenized_sequences)

            # 2. Repeat tokens
            repeated_tokens = tokenized_sequences * self.n_repetitions

            # 3. Mask tokens
            masked_tokens = map(mask_tokens, repeated_tokens)

            # # 4. Padd
            padded_tokens = map(self.padd_masked_data, masked_tokens)
            padded_tokens = list(padded_tokens)
            self.unencoded_data = padded_tokens
            # padded_tokens = masked_tokens
            # self.unencoded_data = masked_tokens

            # 5. Encode
            encoded_data = map(lambda data: (tokenizer.encode_tokens(data[0]),
                                             tokenizer.encode_tokens(data[1]),
                                             data[2]),
                               padded_tokens)

            # 6. Transform to torch Tensor
            tensor_data = map(lambda data: (torch.LongTensor(data[0]),
                                            torch.LongTensor(data[1]),
                                            torch.BoolTensor(data[2])),
                              encoded_data)
            data = tensor_data

        self.data = list(data)
        self.data_length = len(self.data)

        self.sequence_ids = None
        self.distance_matrix = None
        # self.distributions = None
        self.distributions_cdf = None
        if triplets and not classic_triplets:
            self.sequence_ids = {seq: iD for iD, seq in enumerate(self.data)}
            self.distance_matrix = self.derive_distance_matrix()
            self.distributions_cdf = self.derive_distributions_cdf()
            # self.select_negative(anchor=self.data[0])
        pass

    def __len__(self):
        if self.online_masking:
            return self.data_length * self.n_repetitions
        else:
            return self.data_length

    def __getitem__(self, idx):
        if self.triplets:
            positive = self.data[idx % len(self.data)]
            prepared_positive = self.prepare_sequence(positive)
            if not self.classic_triplets:
                negative = self.select_negative(positive)
                prepared_negative = self.prepare_sequence(negative)
                return prepared_positive, prepared_negative
            return prepared_positive

        if self.online_masking:
            sequence = self.data[idx % len(self.data)]
            masked_tokens, tokens, mask_indices = self.prepare_sequence(sequence)
        else:
            masked_tokens, tokens, mask_indices = self.data[idx]
        return masked_tokens, tokens, mask_indices

    def prepare_sequence(self, sequence):
        tokenized_sequence = self.tokenizer.tokenize(sequence)
        masking_result = mask_tokens(self.tokenizer, tokenized_sequence)
        padded_masked_tokens, padded_tokens, padded_mask = self.padd_masked_data(masking_result)
        encoded_tokens = self.tokenizer.encode_tokens(padded_tokens)
        encoded_masked_tokens = self.tokenizer.encode_tokens(padded_masked_tokens)
        masked_tokens = torch.LongTensor(encoded_masked_tokens)
        tokens = torch.LongTensor(encoded_tokens)
        mask_indices = torch.BoolTensor(padded_mask)
        return masked_tokens, tokens, mask_indices

    def padd_masked_data(self, data):
        def pad_seq(seq, prefix, postfix, padding):
            new_seq = prefix
            new_seq += seq
            new_seq += postfix
            new_seq += padding * (self.max_input_length - len(seq) - 2)

            return new_seq

        masked_tokens, tokens, mask = data
        padded_masked_tokens = pad_seq(seq=masked_tokens,
                                       # prefix=[tokenization.REGRESSION_TOKEN],
                                       prefix=[tokenization.CLASSIFIER_TOKEN],
                                       postfix=[tokenization.SEPARATOR_TOKEN],
                                       padding=[tokenization.PADDING_TOKEN]
                                       )

        padded_tokens = pad_seq(seq=tokens,
                                # prefix=[tokenization.REGRESSION_TOKEN],
                                prefix=[tokenization.CLASSIFIER_TOKEN],
                                postfix=[tokenization.SEPARATOR_TOKEN],
                                padding=[tokenization.PADDING_TOKEN]
                                )

        padded_mask = pad_seq(seq=list(mask),
                              prefix=[0],
                              postfix=[0],
                              padding=[0])

        return padded_masked_tokens, padded_tokens, padded_mask

    def derive_distance_matrix(self):
        cache = {}
        changed_cache = False
        if os.path.exists(my_setup.DISTANCE_MATRIX_CACHE):
            print("Using Cached Distance Matrix Information if entry is present")
            with open(my_setup.DISTANCE_MATRIX_CACHE, "r") as file:
                cache = json.load(file)

        distance_mat = np.empty((len(self.data), len(self.data)))
        for iA, anchor in enumerate(self.data):
            if anchor not in cache:
                cache[anchor] = {}

            anchor_cache = cache[anchor]
            for iN, negative in enumerate(self.data):
                # https://pypi.org/project/Levenshtein/
                if negative not in anchor_cache:
                    dist = Levenshtein.distance(anchor, negative) if iA != iN else 0
                    anchor_cache[negative] = dist
                    changed_cache = True

                distance_mat[iA, iN] = anchor_cache[negative]

        if changed_cache:
            with open(my_setup.DISTANCE_MATRIX_CACHE, "w") as file:
                json.dump(cache, file, indent=4)

        return distance_mat

    def derive_distributions_cdf(self):
        distributions_cdf = np.empty((len(self.data), len(self.data)))

        for iA in range(len(self.data)):
            relevant_distances = self.distance_matrix[iA]
            weights = np.exp(-relevant_distances)
            weights[iA] = 0
            distribution = weights / np.sum(weights)
            cumsum = np.cumsum(distribution)
            distributions_cdf[iA] = cumsum

        return distributions_cdf

    def select_negative(self, anchor):
        """

        :param anchor: The sequence acting as anchor
        :return: Another sequence from the dataset, which is drawn according to the distance to the anchor sequence. Closer sequences are preferable.
        """

        anchor_id = self.sequence_ids[anchor]
        cdf = self.distributions_cdf[anchor_id]

        rand_val = np.random.rand()

        # np.searchsorted finds the position where rand_val can be inserted into cdf without breaking the ascending order.
        # As alternative formulation, it returns the first position in cdf, which is larger than rand_val.
        sample_id = np.searchsorted(cdf, rand_val, side="left") - 1

        negative = self.data[sample_id]
        return negative
