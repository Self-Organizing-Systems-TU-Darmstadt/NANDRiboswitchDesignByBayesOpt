import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import my_setup
from data_processing.sequence_utils import prepare_sequence
from data_processing.tokenization import SEPARATOR_TOKEN, PADDING_TOKEN, CLASSIFIER_TOKEN


class Measurements:

    def __init__(self, measurements_file_path, constructs_file_path, limit=None):
        assert measurements_file_path is not None and constructs_file_path is not None


        measurements = pd.read_excel(measurements_file_path)
        constructs = pd.read_excel(constructs_file_path)

        constructs_lookup = dict(constructs.values)

        measurements["Sequence"] = measurements["Construct"].map(lambda const: constructs_lookup[const])

        measurements["Sequence"] = measurements["Sequence"].map(prepare_sequence)

        measurements["Construct"] = measurements["Construct"].map(
            lambda const: const.replace("TN", "TN0") if len(const) == 4 else const)

        rescaler = lambda val: val / 100
        for ligand_comb in my_setup.LIGAND_COMBINATIONS:
            measurements[ligand_comb] = measurements[ligand_comb].map(rescaler)

        if my_setup.PREDICT_SCORE:
            measurements["Score"] = [my_setup.SCORE_FUNC(measurements.iloc[iRow]) for iRow in range(len(measurements))]

        data = measurements
        data = data.sort_values(by="Construct")

        if limit is not None and type(limit) is int and limit < len(data):
            data = data[:limit]

        self.data = data
        pass

    def split(self, split=None, split_by_construct=True, shuffle=True):
        if split is None:
            split = [0.7, 0.15, 0.15]

        validation = []

        data_to_split = self.data
        if split_by_construct:
            data_to_split = set(self.data["Construct"])
            data_to_split = [construct for construct in self.data["Construct"] if construct in data_to_split]

        training, test = train_test_split(data_to_split, train_size=split[0], shuffle=shuffle)
        if len(split) == 3:
            validation, test = train_test_split(test, test_size=split[2] / (split[1] + split[2]), shuffle=shuffle)

        if split_by_construct:
            training = self.data[self.data["Construct"].map(lambda const: const in training)]
            validation = self.data[self.data["Construct"].map(lambda const: const in validation)]
            test = self.data[self.data["Construct"].map(lambda const: const in test)]

            if shuffle:
                training = training.sample(len(training))
                validation = validation.sample(len(validation))
                test = test.sample(len(test))

        if len(split) == 3:
            return training, validation, test
        else:
            return training, test


class SequenceToExpressionDataset(Dataset):
    def __init__(self, measurements: pd.DataFrame, max_sequence_length=-1, tokenizer=None):
        if max_sequence_length < 0:
            max_sequence_length = my_setup.MAX_SEQUENCE_LENGTH

        self.input_sequence_length = max_sequence_length + 2
        if tokenizer is None:
            tokenizer = my_setup.TOKENIZER

        data = []

        seq_only = True  # if seq_only = False, the input to the net is the sequence and the ligand condition. Otherwise it is only the sequence
        self.seq_only = seq_only

        if seq_only:
            if my_setup.PREDICT_SCORE:
                resulting_data = measurements[["Sequence", "Score"]].values
                # resulting_data = map(lambda elem: (elem[0], [elem[1]]), resulting_data)
            else:
                resulting_data = measurements[["Sequence"] + my_setup.LIGAND_COMBINATIONS].values
            resulting_data = list(map(lambda elem: [elem[0], tuple(elem[1:])], resulting_data))
            data = resulting_data
        else:
            for iM in range(len(measurements)):
                measurement = measurements.iloc[iM]

                resulting_data = self._process_measurement(measurement)

                data += resulting_data

        # 1. Tokenize
        # 2. Padd
        # 3. Encode Tokens
        # 4. Transform to torch Tensor

        # 1. Tokenize

        tokenized_sequences = map(lambda val: [tokenizer.tokenize(val[0].upper())] + val[1:], data)
        # tokenized_sequences = map(lambda val: (tokenizer.tokenize(val[0].upper()), (val[1], val[2]), val[3]),data)
        tokenized_sequences = list(tokenized_sequences)

        # 2. Padd
        padded_data = map(lambda entry: [self.prepare_token_seq(entry[0])] + entry[1:],
                          tokenized_sequences)
        padded_data = list(padded_data)
        self.unencoded_data = padded_data

        # 3. Encode
        encoded_data = map(lambda entry: [tokenizer.encode_tokens(entry[0])] + entry[1:],
                           padded_data)

        # 4. Transform to torch Tensor
        tensor_data = map(lambda entry: [torch.LongTensor(entry[0])] + [torch.FloatTensor(ent) for ent in entry[1:]],
                          encoded_data)

        self.data = list(tensor_data)
        self.data_length = len(self.data)

        pass

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # The current item is created on the fly
        cur_item = self.data[idx]

        if self.seq_only:
            x, y = cur_item
        else:
            seq, ligand_combination, val = cur_item
            x = (seq, ligand_combination)
            y = val
        return x, y

    def _process_measurement(self, measurement):
        # Extract the measurement values for each condition and transform the conditions into a two values.

        # Yields four entries for every existing entry

        # Sequence
        # Tc
        # Neo
        # expression level
        data = [None] * 4
        ligand_combinations = [["w/o", "Tc"], ["Neo", "both"]]
        for iNeo in range(2):
            for iTc in range(2):
                ligand_combination = ligand_combinations[iNeo][iTc]
                data[iNeo * 2 + iTc] = [measurement["Sequence"],
                                        (iNeo, iTc),
                                        (measurement[ligand_combination],)]

        return data

    def _process_measurement_multi(self, measurement):
        # Sequence to expression with all combinations as output

        ligand_combinations = my_setup.LIGAND_COMBINATIONS

        if my_setup.PREDICT_SCORE:
            data = [measurement["Sequence"]] + [tuple([measurement["Score"]])]
        else:
            data = [measurement["Sequence"]] + [tuple([measurement[ligand_combination]
                                                       for ligand_combination in ligand_combinations])]
        return [data]

    def prepare_token_seq(self, token_sequence):
        def pad_seq(seq, prefix, postfix, padding):
            new_seq = prefix
            new_seq += seq
            new_seq += postfix
            new_seq += padding * (self.input_sequence_length - len(seq) - 2)

            return new_seq

        padded_token_sequence = pad_seq(seq=token_sequence,
                                        prefix=[CLASSIFIER_TOKEN],
                                        # Should actually be REGRESSION_TOKEN. However, dnabert only supports for Classifier Token
                                        postfix=[SEPARATOR_TOKEN],
                                        padding=[PADDING_TOKEN]
                                        )

        return padded_token_sequence


if __name__ == '__main__':
    constructs_file_path = "../_data/Constructs_first_run.xlsx"
    measurements_file_path = "../_data/Dataset_Mean_first_run.xlsx"
    # measurements_file_path = "../_data/Dataset.xlsx"

    measurements = Measurements(measurements_file_path=measurements_file_path,
                                constructs_file_path=constructs_file_path)

    splitted_measurements = measurements.split([0.7, 0.15, 0.15])

    training_data = SequenceToExpressionDataset(measurements=splitted_measurements[0])
    pass
