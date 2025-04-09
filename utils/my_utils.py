import os

import numpy as np

def short_summary(model):
    NUM_PARAMETERS = {"Not Trainable": 0, "Trainable": 0, "Overall": 0}
    for params in model.parameters():
        num_params = params.numel()
        trainable = params.requires_grad
        key = list(NUM_PARAMETERS.keys())[trainable]
        NUM_PARAMETERS[key] += num_params
        NUM_PARAMETERS["Overall"] += num_params

    print(NUM_PARAMETERS)


def to_numpy(tensor):
    return tensor.detach().numpy()


def get_best_params_path(dir, identifier):
    current_model_params = []
    cur_dir = dir + "/" if not dir.endswith("/") else dir
    for elem in os.listdir(cur_dir):
        if identifier is None or identifier in elem:
            current_model_params.append(elem)

    filenames = [os.path.splitext(elem)[0] for elem in current_model_params]
    losses = [float(name.split("=")[-1]) for name in filenames]
    if len(losses) == 0:
        return None
    id = np.argmax(losses)
    param_file = current_model_params[id]
    param_file_path = cur_dir + param_file

    return param_file_path
