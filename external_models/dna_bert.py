import torch
from transformers import AutoTokenizer


class DNABertTokenizer:
    # The Tokenizer actually performs the encoding
    def __init__(self, k=3):
        super(DNABertTokenizer, self).__init__()
        self.k = k

        if k not in [3, 4, 5, 6]:
            raise Exception(
                f"There does not exist a pretrained DNABERT  Model for {k}. k needs to be one of [3, 4, 5, 6].")
        self.tokenizer = AutoTokenizer.from_pretrained(f"zhihan1996/DNA_bert_{k}")

        self.VOCABULARY = self.tokenizer.vocab

        self.BASE_TOKENS = list(set(self.tokenizer.vocab.keys()).difference(set(self.tokenizer.all_special_tokens)))
        self.BASE_TOKENS_VOCABULARY = {token: self.VOCABULARY[token] for token in self.BASE_TOKENS}
        self.BASE_TOKENS_IDs = torch.tensor(list(self.BASE_TOKENS_VOCABULARY.values()), dtype=torch.long)

        self.VOCABULARY_SIZE = len(self.VOCABULARY)

        self.PAD_TOKEN = self.tokenizer.pad_token
        self.PAD_TOKEN_ID = self.VOCABULARY[self.tokenizer.pad_token]

        pass

    def encode(self, sequence):
        tokenized_sequence = self.tokenize(sequence)
        token_string = " ".join(tokenized_sequence)
        # encoded_sequence = self.encode_tokens(tokenized_sequence)
        encoded_sequence = self.encode_token_string(token_string)
        return encoded_sequence

    def encode_tokens(self, tokens):
        encoding = [self.VOCABULARY[token] for token in tokens]
        return encoding

    def encode_token_string(self, token_string):
        encoded_token_string = self.tokenizer(token_string)
        return encoded_token_string

    def tokenize(self, sequence):
        num_kmers = len(sequence) - self.k + 1
        tokens = [None] * num_kmers
        for iS in range(num_kmers):
            kmer = sequence[iS:iS + self.k]
            tokens[iS] = kmer

        return tokens

    def decode_logits(self, logits):
        if len(logits.size()) == 2:
            cur_logits = logits.unsqeeze(0)  # Add Batch Dimension

        distribution = torch.softmax(cur_logits, dim=2)

        ids = torch.argmax(distribution, dim=2)

        elems = self.decode(ids)
        return elems

    def decode(self, output_ids):
        vocab = self.tokenizer.vocab
        decoding_vocab = {vocab[key]: key for key in vocab}

        elems = []
        for elem in output_ids:
            decoded_elem = []
            for cur_id in elem:
                decoded_elem.append(decoding_vocab[int(cur_id)])

            decoded_elem = " ".join(decoded_elem)
            elems.append(decoded_elem)

        return elems