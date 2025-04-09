import itertools

import torch

VALID_BASES = ["A", "G", "C", "T"]

UNKNOWN_TOKEN = "[UNKNW]"
MASK_TOKEN = "[MASK]"
PADDING_TOKEN = "[PAD]"
CLASSIFIER_TOKEN = "[CLS]"
REGRESSION_TOKEN = "[REG]"
SEPARATOR_TOKEN = "[SEP]"

SPECIAL_TOKENS = [UNKNOWN_TOKEN, MASK_TOKEN, PADDING_TOKEN, CLASSIFIER_TOKEN, REGRESSION_TOKEN, SEPARATOR_TOKEN]


class Tokenizer:

    def __init__(self, tuple_size=1):
        self.tuple_size = tuple_size

        self.BASE_TOKENS = ["".join(elem) for elem in itertools.product(*[VALID_BASES] * tuple_size)]

        self.TOKENS = self.BASE_TOKENS + SPECIAL_TOKENS

        self.VOCABULARY = {token: id for id, token in enumerate(self.TOKENS)}

        self.BASE_TOKENS_VOCABULARY = {token: self.VOCABULARY[token] for token in self.BASE_TOKENS}
        self.BASE_TOKENS_IDs = torch.tensor(self.BASE_TOKENS_VOCABULARY.values())

        self.VOCABULARY_SIZE = len(self.VOCABULARY)

    def tokenize(self, sequence):
        if len(sequence) % self.tuple_size != 0:
            raise Exception(f"Sequence length not divisible by tuple size {self.tuple_size}")

        num_tokens = int(len(sequence) / self.tuple_size)

        tokens = [None] * num_tokens

        for iT in range(num_tokens):
            tokens[iT] = sequence[iT * self.tuple_size: (iT + 1) * self.tuple_size]

        return tokens

    def encode_tokens(self, tokens):

        encoding = [self.VOCABULARY[token] for token in tokens]
        return encoding

    def encode(self, sequence):
        tokenized_sequence = self.tokenize(sequence)
        encoded_sequence = self.encode_tokens(tokenized_sequence)
        return encoded_sequence


class KmerTokenizer(Tokenizer):
    def __init__(self, k=2):
        super().__init__(tuple_size=k)

        self.k_mer = k

    def tokenize(self, sequence):
        num_kmers = len(sequence) - self.k_mer + 1
        tokens = [None] * num_kmers
        for iS in range(num_kmers):
            kmer = sequence[iS:iS + self.k_mer]
            tokens[iS] = kmer

        return tokens


class CodonTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(tuple_size=3)


if __name__ == '__main__':
    tokenizer = KmerTokenizer(k=3)
    sequence = "AGCTCCCGGGAGGT"

    encoded_sequence = tokenizer.encode(sequence=sequence)
    print("Encoded Sequence")
    print(encoded_sequence)
    pass
