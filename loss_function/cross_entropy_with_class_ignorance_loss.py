import torch
from transformers.modeling_outputs import MaskedLMOutput


def masked_cross_entropy_loss(logits, labels, masks):

    cur_labels = labels
    cur_logits = logits
    if isinstance(logits, MaskedLMOutput): # Adaption to directly support DNABERT
        cur_logits = logits[0]
    relevant_logits = cur_logits[masks] # Requires logits to be in the shape N x L x C (with N Batch Size, L sequence Length, and C num classes)
    relevant_labels = cur_labels[masks] # Requires labels to be in the shape N x L with values within [0, ..., C-1]
    avg_loss = torch.nn.functional.cross_entropy(input=relevant_logits, target=relevant_labels, reduction='mean')

    return avg_loss
