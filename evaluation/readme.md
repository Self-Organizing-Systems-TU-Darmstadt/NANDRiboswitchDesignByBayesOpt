# Setup

Clone evaluation branch of repository 
```
git clone -b evaluation https://github.com/Self-Organizing-Systems-TU-Darmstadt/NANDRiboswitchDesignByBayesOpt.git
cd NANDRiboswitchDesignByBayesOpt/
```


Create a conda venv for the project
```
conda create -n NANDRiboswitchDesignByBayesOpt python=3.11
conda activate NANDRiboswitchDesignByBayesOpt
```

Install the torch version with GPU support for your system from the official website. The external model used requires GPU support. The installer used for linux is
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Install requirements for the NANDRiboswitchDesignByBayesOpt project:
```
pip install -r requirements.txt
```

## Extended setup for evaluation
The evaluation uses a large promoter dataset of [Gosai *et al.* 2024](https://www.nature.com/articles/s41586-024-08070-z#code-availability) which can be downloaded as [Supplementary Table 2](git clone https://github.com/sjgosai/boda2.git). To match the dataset, we also use their model (`boda2`), which is available at https://github.com/sjgosai/boda2/tree/main. The following steps describe how to download their model and integrate it into our framework, while we assume that you followed the previous setup and are currently in the directory `NANDRiboswitchDesignByBayesOpt/`.

The setup for boda2 follows the prescriptions in their GitHub. Please also consider their GitHub regarding the compatibility with your system. The evaluation was executed on a multi GPU system featuring `Nvidia A100 80GB` GPUs (actual memory usage was below 10GB).

```
git clone https://github.com/sjgosai/boda2.git
cd boda2/

pip install --upgrade pip==21.3.1
pip install --no-cache-dir -r requirements.txt
pip install -e .
```

Now please download the dataset file (Supplementary Table 2) from the paper's website or the direct link:
```
https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08070-z/MediaObjects/41586_2024_8070_MOESM4_ESM.txt
```
Move again into `NANDRiboswitchDesignByBayesOpt/`by executing
```
cd ..
```
Create the new directory `_data_evaluation/CRE/`inside `NANDRiboswitchDesignByBayesOpt/` and upload the downloaded dataset into this directory $\rightarrow$ `NANDRiboswitchDesignByBayesOpt/_data_evaluation/CRE/41586_2024_8070_MOESM4_ESM.txt`

Next we download the parent weights required for training the model:
```
gsutil cp gs://tewhey-public-data/CODA_resources/my-model.epoch_5-step_19885.pkl ./
```
Also move the downloaded file into the directory `CRE/`such that the path is:
```
NANDRiboswitchDesignByBayesOpt/_data_evaluation/CRE/my-model.epoch_5-step_19885.pkl
```

This completes the setup for the evaluation.

# Run Evaluation
All evaluation related files are located in `NANDRiboswitchDesignByBayesOpt/evaluation/`. `bayesian_optimization_CRE.py` implements a single round of Bayesian optimization with the ensemble model being based on `boda2`. Except for modifications to the new task, the Bayesian optimization setting follows the same as employed for the NAND hybrid riboswitch design. The actual evaluation is done by `bayes_CRE_evaluator.py`, which simulates the experimental steps by looking up the true data from the dataset and preparing everything for the next round of Bayesian optimization.

Navigate into the evaluation directory (serves as working directory).

```
cd evaluation/
```
Execute the code for the evaluation.
```
PYTHONPATH=../ python bayes_CRE_evaluator.py -n 10
```
