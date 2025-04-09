import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import my_setup


class BatchAcquisitionFunction:
    def __init__(self, model, model_args, domain, data, model_output_transform=None, config=None):
        self.label = "BatchAcquisitionFunction"
        self.model = model
        self.model_args = model_args
        self.domain = domain
        self.measurements = data
        self.measurements.data["Type"] = "Measurement"
        self.measurements.data["Score"] = np.nan
        self.measurements.data["Utility Value"] = np.nan
        self.model_output_transform = model_output_transform
        self.config = config
        self.log_file = sys.stdout
        self.log_file_path = None
        if "log_dir" in config:
            self.log_file_path = config["log_dir"] + "log_file.txt"
            self.log_file = open(self.log_file_path, "w")
        pass

    def utility_function(self):
        """
        This method shall apply the model to each value in the domain.
        From the model, the desired values determining the utility function shall be derived.
        :return:
        """
        raise Exception("The utility function needs to be implemented by the sub class.")

    def update_model(self, new_entry=None, fit_model=True, *args, **kwargs):
        if new_entry is not None:
            information_to_add = {key: new_entry[key] for key in new_entry if not ("raw" in key or "prediction" in key)}
            self.measurements.data.loc[len(self.measurements.data)] = pd.Series(information_to_add)
            print("\nAdded the following entry to the model:", file=self.log_file)
            print(information_to_add, file=self.log_file)
            print("\n", file=self.log_file)
            self.log_file.flush()

        if fit_model:
            self.model.fit(measurements=self.measurements, **self.model_args)

        return

    def __call__(self, batch_size=1):
        domain = self.domain
        proposals = {}
        iS = 0
        self.update_model(new_entry=None, fit_model=True)
        while len(proposals) < batch_size and iS < 5 * batch_size:
            pred, u_func_vals, model_outputs = self.utility_function()
            # Performance could be significantly increased for Thomposon Sampling by deviating from the presented order
            # (all samples could be derived with a single evaluation of the model on the domain)

            # Set UCB of already selected proposals to -infty to prevent infinite loops
            # masked_vals = list(map(lambda elem: elem in proposals.keys(), domain[0]))
            # u_func_vals[masked_vals] = -np.infty
            print(f"Iteration {iS}:", file=self.log_file)
            searching_proposal = True
            while searching_proposal:
                proposed_index = np.argmax(u_func_vals)
                proposal = domain[0][proposed_index], domain[1][proposed_index]
                seq = proposal[0]
                tokens = proposal[1]
                utility_value = u_func_vals[proposed_index]
                value = pred[proposed_index]
                print(f"Proposed: {seq} (value = {value}, utility_value = {utility_value})", file=self.log_file)

                proposal_dict = {"Sequence": seq,
                                 "Score": float(value),
                                 "Type": "Prediction",
                                 "Utility Value": float(utility_value),
                                 # "Tokens": tokens
                                 }
                if not my_setup.PREDICT_SCORE:
                    for iC, combi in enumerate(my_setup.LIGAND_COMBINATIONS):
                        cur_vals = model_outputs[proposed_index][iC]
                        proposal_dict[combi + "_predictions"] = list(cur_vals)
                        proposal_dict[combi] = np.mean(cur_vals, axis=-1)
                is_in_proposals = seq in proposals
                is_in_measurements = seq in self.measurements.data["Sequence"].values
                if is_in_proposals or is_in_measurements:
                    ident_str = "proposals" if is_in_proposals else ("measurements" if is_in_measurements else "ERROR")
                    print("Proposed sequence is in %s and therefore discarded." % ident_str, file=self.log_file)
                    print("Proposal is:", proposal_dict, file=self.log_file)
                    print("", file=self.log_file)
                    u_func_vals[proposed_index] = -np.infty
                    pass
                else:
                    searching_proposal = False

            new_entry = None
            self.log_file.flush()
            if seq not in proposals and seq not in self.measurements.data["Sequence"].values:
                proposals[seq] = proposal_dict
                new_entry = proposal_dict
            # The model is retrained after each iteration to prevent the bayesian optimization getting stuck
            if len(proposals) <= batch_size:
                self.update_model(new_entry, fit_model=not (len(proposals) == batch_size))

            iS += 1
        return proposals


class UpperConfidenceBound(BatchAcquisitionFunction):
    def __init__(self, model, model_args, domain, data, model_output_transform=None, config=None):
        super().__init__(model, model_args, domain, data, model_output_transform=model_output_transform, config=config)
        self.label = "UpperConfidenceBound"
        self.coverage_probability = config["bayesian_optimization"]["coverage_probability"]
        self.normal_approximation_mode = config["bayesian_optimization"]["normal_approximation_mode"]
        self.beta = norm().ppf(self.coverage_probability)
        self.std_dev_scaler = 1
        pass

    def utility_function(self):
        inputs = self.domain[1]
        # Possibly perform this step batch wise as the duration is very long.
        num_elems = len(inputs)

        inputs = inputs.to(my_setup.DEVICE_TRAINING)
        self.model = self.model.to(my_setup.DEVICE_TRAINING)
        batch_size = 16

        num_batches = int(np.ceil(num_elems / batch_size))
        start = time.time()
        model_outputs = [None] * num_batches
        for iB in range(num_batches):
            start_index = batch_size * iB
            end_index = batch_size * (iB + 1)
            model_output = self.model(inputs[start_index: end_index], combine_outputs=False)
            if hasattr(model_output, "detach"):
                model_output = model_output.detach()
            model_output = model_output.to("cpu")
            model_output = np.array(model_output, dtype=float)
            model_outputs[iB] = model_output

        model_outputs = np.concatenate(model_outputs, axis=0)
        end = time.time()
        duration = end - start
        print(f"Model Evaluation for {len(inputs)} entries took {duration} s ({duration / len(inputs)} s per sample)",
              file=self.log_file)

        transformed_output = model_outputs
        # In case the output is the expression level, transform to the score first and derive then the UCB
        if self.model_output_transform is not None:
            transformed_output = self.model_output_transform(model_outputs)
        # transformed_output is of shape M x 1 x N with M the number of sequences and N the ensemble size

        means = np.mean(transformed_output, axis=-1)
        if self.normal_approximation_mode:
            # Determination of the UCB via a normal distribution approximation of the data

            std_devs = np.std(transformed_output, axis=-1)
            means = means[:, 0]
            std_devs = std_devs[:, 0]
            ucb = means + self.beta * self.std_dev_scaler * std_devs
        else:
            ensemble_size = transformed_output.shape[-1]
            index = ensemble_size * self.coverage_probability
            index = index - 1  # Shift to 0 to N-1 indexing
            if index - int(index) == 0:
                # In case the index is an integer value, take the average of the current and next
                index = int(index)
                indexes = [index, index + 1]
            else:
                indexes = [int(np.ceil(index))]  # Round up to the next full Integer

            sorted_output = np.sort(transformed_output, axis=-1)
            ucb = np.mean(sorted_output[:, 0, indexes], axis=-1, dtype=float)

        return means, ucb, model_outputs


class ThompsonSampling(BatchAcquisitionFunction):
    def __init__(self, model, model_args, domain, data, model_output_transform=None, config=None):
        super().__init__(model, model_args, domain, data, model_output_transform=model_output_transform, config=config)
        self.label = "ThompsonSampling"

        pass

    def utility_function(self):
        inputs = self.domain[1]
        model_outputs = self.model(inputs)
        if hasattr(model_outputs, "detach"):
            model_outputs = model_outputs.detach()
        model_outputs = np.array(model_outputs)

        transformed_output = model_outputs
        # In case the output is the expression level, transform to the score first and derive then the UCB
        if self.model_output_transform is not None:
            transformed_output = self.model_output_transform(model_outputs)

        means = np.mean(model_outputs, axis=-1)
        # std_devs = np.std(model_outputs, axis=-1)

        ensemble_size = model_outputs.shape[-1]
        function_id = np.random.randint(0, ensemble_size)

        function_outputs = model_outputs[:, 0, function_id]
        return means, function_outputs, model_outputs

    def update_model(self, *args, **kwargs):
        # One could remove the sampled function from the pool of sampled functions.
        # However, this is not required for the correct functioning of this approach.
        return
