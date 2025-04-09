import torch
import re

import my_setup
from data_processing.tokenization import MASK_TOKEN


def prepare_sequence(seq):
    if my_setup.EXTRACT_RANDOMISATION:
        new_seq = seq if len(seq) <= 14 else extract_randomisation(seq)
    elif my_setup.EXTRACT_CONSTRUCT:
        new_seq = seq if len(seq) <= 70 else extract_construct(seq)
    else:
        new_seq = str(seq)

    return new_seq


def extract_randomisation(seq, split=False, exact_mode=True):
    randomization = extract_construct(seq, extract_randomization=1 if split == False else 2, exact_mode=exact_mode)
    return randomization



def extract_construct(seq, exact_mode=True, extract_randomization=0):
    res = identify_template(seq, exact_mode, return_match=True)
    if res is None:
        return None

    T_ID, match = res

    randomizations_length = {"T1": 5, "T2": 6, "T3": 7, "T4": 7}

    test_seq = seq.upper()
    construct = test_seq[match.start():match.end()]
    if extract_randomization > 0:
        starts = [1, len(construct) - 1 - randomizations_length[T_ID]]
        randomizations = [construct[start:start + randomizations_length[T_ID]] for start in starts]
        if extract_randomization == 2:
            return randomizations
        return "".join(randomizations)
    return construct


def identify_template(seq, exact_mode=True, return_match=False):
    if exact_mode:
        subconstruct = "AAACATACTCGCTTGTCCTTTAATGGTCCTTGAGAGGTGAAGAATACGACCACC"
        ref_seqs = [subconstruct]
        T1 = "TNNNNNAAACATACTCGCTTGTCCTTTAATGGTCCTTGAGAGGTGAAGAATACGACCACCNNNNNA"
        T2 = "TNNNNNCAAACATACTCGCTTGTCCTTTAATGGTCCTTGAGAGGTGAAGAATACGACCACCGNNNNNA"
        T3 = "TNNNNNCCAAACATACTCGCTTGTCCTTTAATGGTCCTTGAGAGGTGAAGAATACGACCACCGGNNNNNA"
        T4 = "TNNNNNGCAAACATACTCGCTTGTCCTTTAATGGTCCTTGAGAGGTGAAGAATACGACCACCGCNNNNNA"
    else:
        ref_seqs = ["AAACATAC",
                    "CCACC"]
        T1 = "TNNNNNAAACATAC" + r"[AGCT]{37,41}" + "GACCACCNNNNNA"
        T2 = "TNNNNNCAAACATAC" + r"[AGCT]{37,41}" + "GACCACCGNNNNNA"
        T3 = "TNNNNNCCAAACATAC" + r"[AGCT]{37,41}" + "GACCACCGGNNNNNA"
        T4 = "TNNNNNGCAAACATAC" + r"[AGCT]{37,41}" + "GACCACCGCNNNNNA"

    # The order is of importance as T1 is potentially also a match for the matches of T2 and so on.
    T = [T4, T3, T2, T1]
    T_IDs = ["T4", "T3", "T2", "T1"]

    T = [t.replace("N", ".") for t in T]

    test_seq = seq.upper()
    # is_construct = subconstruct in test_seq
    indexes = [-1] * len(ref_seqs)
    for iR, ref_seq in enumerate(ref_seqs):
        start = 0 if iR == 0 else indexes[iR - 1]
        index = test_seq[start:].find(ref_seq)
        indexes[iR] = index

    is_construct = all([i >= 0 for i in indexes])

    if is_construct:

        for iT, t in enumerate(T):
            match = re.search(t, test_seq)
            if match is not None:
                if return_match:
                    return T_IDs[iT], match
                return T_IDs[iT]
    return None


def mask_tokens(tokenizer, tokens):
    mask_prob = 0.15
    num_base_tokens = len(tokenizer.BASE_TOKENS)


    masked_tokens = list(tokens)
    while True:
        rand_vals = torch.rand(size=(len(tokens),))
        mask_indices: bool = rand_vals < mask_prob

        non_zero_vals = torch.count_nonzero(mask_indices)
        if non_zero_vals == 0 or non_zero_vals == len(tokens):
            continue

        mask_replace_with_mask_token = rand_vals < mask_prob * 0.8
        mask_replace_with_random_token = torch.logical_and(rand_vals >= mask_prob * 0.8,
                                                           rand_vals < mask_prob * 0.9)

        random_replacement = torch.empty(mask_replace_with_random_token.size(), dtype=torch.int64)
        rand_vals = torch.randint(low=0, high=num_base_tokens, dtype=torch.int64,
                                  size=(torch.count_nonzero(mask_replace_with_random_token).item(),))
        random_replacement[mask_replace_with_random_token] = rand_vals

        break

    for iP in torch.where(mask_replace_with_mask_token == 1)[0]:
        masked_tokens[iP] = MASK_TOKEN
    for iP in torch.where(mask_replace_with_random_token == 1)[0]:
        masked_tokens[iP] = tokenizer.BASE_TOKENS[random_replacement[iP]]

    return masked_tokens, tokens, mask_indices
