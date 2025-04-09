import os
import torch
import yaml

from external_models.dna_bert import DNABertTokenizer

DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu")

DEVICE = "cpu"

DEVICE_TRAINING = ("cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else
                   "cpu")

if DEVICE_TRAINING == "mps":
    DEVICE_TRAINING = "cpu"

DEVICE_TRAINING = "cpu"



CWD = os.getcwd()




ALLOWED_BASES = set("AGCT")
VOCABULARY = {c: iC for iC, c in enumerate(ALLOWED_BASES)}  # Add your custom int mapping here
DECODE_TO_VOCABULARY = {VOCABULARY[c]: c for c in VOCABULARY}

PIN_MEMORY = True if DEVICE == "cpu" and DEVICE_TRAINING != "cpu" else False


TOKENIZER = DNABertTokenizer()
VOCABULARY_SIZE = len(VOCABULARY)
SPECIAL_CHARACTERS_VOCAB_SIZE = 1

MASK_TOKEN_ID = VOCABULARY_SIZE
IGNORE_CLASS_ID = -1

NUMBER_OF_TIMES_SEQUENCE_IS_REPEATED = 50

MODEL_SAVE_POINTS_DIRECTORY = "model_savepoints/"

DATA_DIRECTORY = "_data/"
MODEL_PARAMETERS_DIRECTORY = "model_parameters/"
MODEL_REGRESSION_PARAMETERS_DIRECTORY = "model_regression_parameters/"

LIGAND_COMBINATIONS = ["w/o", "Tc", "Neo", "both"]


def SCORE_FUNC(entry):
    ON_STATE_VALS = ["w/o", "Tc", "Neo"]
    OFF_STATE_VALS = ["both"]
    on_state_vals = [entry[id] for id in ON_STATE_VALS]
    off_state_vals = [entry[id] for id in OFF_STATE_VALS]
    score = min(on_state_vals) / max(off_state_vals)
    return score


DIRECTORIES = [DATA_DIRECTORY, MODEL_PARAMETERS_DIRECTORY, MODEL_SAVE_POINTS_DIRECTORY]

for dir in DIRECTORIES:
    if not os.path.exists(dir):
        os.makedirs(dir)



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

PREDICT_SCORE = bool(config["training_regression"]["predict_score"])
TRIPLET_MODE = bool(config["training_encoder"]["triplet_mode"])
TRIPLET_MODE_CLASSIC_NEGATIVE_SELECTION = bool(config["training_encoder"]["triplet_mode_classic_negative_selection"])
EMBEDDING_DIM = int(config["encoder"]["embedding_dim"])

MAX_SEQUENCE_LENGTH = int(config["general"]["max_sequence_length"])

DATA_DIRECTORY = os.path.join(CWD, config["data"]["data_directory"])
NGS_SEQUENCES_FILE = os.path.join(CWD, config["data"]["ngs_sequences_file"])



EXTRACT_RANDOMISATION = False
if MAX_SEQUENCE_LENGTH <= 14:
    EXTRACT_RANDOMISATION = True

EXTRACT_CONSTRUCT = False
if MAX_SEQUENCE_LENGTH <= 70:
    EXTRACT_CONSTRUCT = True