"""
The task of this file is to split the provided sequences file into training, validation and test data.
"""
import numpy as np

if __name__ == '__main__':
    filename = "ngs_sequences.txt"

    split = [0.7, 0.15, 0.15]

    with open(filename, "r") as file:
        sequences = file.readlines()

    sequences = [seq.replace("\n", "") for seq in sequences if len(seq) < 170]

    num_data = len(sequences)

    val_size = int(np.ceil(num_data * split[1]))
    test_size = int(np.ceil(num_data * split[2]))
    train_size = num_data - val_size - test_size

    np.random.shuffle(sequences)

    train_set = sequences[:train_size]
    val_set = sequences[train_size: train_size + val_size]
    test_set = sequences[train_size + val_size:]
    datasets = train_set, val_set, test_set
    labels = "train", "val", "test"
    if len(test_set) != test_size or len(val_set) != val_size or len(train_set) != train_size:
        raise Exception(f"Error in Dataset size {len(test_set) != test_size} or {len(val_set) != val_size} or {len(train_set) != train_size}")

    for label, dataset in zip(labels, datasets):
        with open(filename.replace(".txt", f"_{label}.txt"), "w") as file:
            file.writelines([seq + "\n" for seq in dataset])
    pass
