# Neural Pattern Learning (NEPAL) Attack

This repository accompanies the paper **“NEPAL: Climbing Toward the Peak of Re-Identification in Privacy-Preserving Record Linkage”**, which introduces the **Neural Pattern Learning (NEPAL) Attack**.  
It provides documentation and resources for reproducing the experiments and analyses presented in the paper.


## Project Overview

The **NEPAL Attack** models a *machine learning–based adversary* that performs re-identification in **Privacy-Preserving Record Linkage (PPRL)** systems based on known plaintext–encoding pairs.  
Unlike traditional **Pattern Mining Attacks (PMAs)** that rely on scheme-specific heuristics, NEPAL formulates pattern mining as a general learning problem. It uses neural networks to learn correlations between encoded records and their underlying plaintext structures, enabling large-scale, scheme-agnostic plaintext reconstruction.

The attack consists of two major stages:

1. **Pattern Mining** – A neural model learns mappings between encodings and their constituent *q-grams* (substrings of the original identifiers). This is framed as a multi-label classification task.
2. **Plaintext Reconstruction** – The predicted q-grams are assembled into complete identifiers using a graph-based reconstruction algorithm.

For a detailed description of the attacker model, theoretical background, and evaluation results, see the paper.


## Getting Started with Docker

The simplest way to reproduce the NEPAL pipeline is to run the implementation repository inside Docker.  
The following setup reproduces the environment used during paper preparation:

```bash
git clone <nepal-repository>
cd <nepal-repository>
git submodule update --init --recursive --remote

docker build -t nepal .
docker run --gpus all -it -v $(pwd):/usr/app nepal bash
```
Note: GPU access is optional but strongly recommended for hyperparameter optimization.
The repository will be mounted inside the container at /usr/app.


## Running the Default NEPAL Case

A default configuration is provided for the NEPAL attack.
Once inside the container, execute:
```bash
python3 main.py --config nepal_config.json
```
This command launches the complete NEPAL pipeline, including:
- data preprocessing,
- neural model training, and
- plaintext reconstruction.

Results are written to the [experiment_results](experiment_results) directory.
See [docs/parameters.md](docs/parameters.md) for a detailed explanation of configuration options and schema.

## Prepare your Dataset
The code expects a tab-separated file with one record per row. The fist row must be a 
header specifying the column names. 
Internally, the values stored in the columns are concatenated according to column ordering and normalized (switch to lowercase, remove whitespace and missing values).
**The last column must contain a unique ID.**

If you have data in `.csv`, `.xls` or `.xlsx` format, you may run ``python preprocessing.py`` for convenient conversion. 
The script will guide you through the process. 

In the `data` directory, this repository already provides datasets which can be used.

## Batch Experiment Setup

To run multiple experiments or reproduce the experiments from the paper, use the experiment script:
```bash
python3 experiment_setup.py
```
This script runs multiple configurations automatically to produce results as in the paper.


## Analysis and Evaluation

The analysis notebook [analysis.ipnb](analysis.ipnb) reproduce the figures reported in the paper.
Open the notebook and ensure that the output file from [extract_nepal_results.py](extract_nepal_results.py) is generated correctly.
[extract_nepal_results.py](extract_nepal_results.py) uses the results produced in [experiment_results](experiment_results)


## Summary of Key Contributions (from the Paper)
### Generalized Pattern Learning:
NEPAL reframes cryptanalysis of similarity-preserving encodings as a supervised learning task, enabling the model to learn directly from encoding–plaintext pairs and generalize across multiple encoding schemes.
### Two-Stage Attack Pipeline 
(1) Pattern Mining using neural networks to predict constituent q-grams from encoded data, and
(2) Plaintext Reconstruction assembling the predicted fragments into complete identifiers.
### Comprehensive Evaluation 
Experiments were conducted on eight datasets (including FakeName, Euro Person, and Titanic), across three encoding schemes:
Bloom Filters (BF), Two-Step Hashing (TSH), and Tabulation MinHash (TMH).
### Performance Highlights 
- Achieved Dice coefficients up to 0.997 (indicating near-perfect q-gram reconstruction).
- Re-identified up to 33.05% of encoded records exactly.
- Demonstrated that TSH and BF are the most vulnerable encoding schemes, while TMH is more resilient.

## Additional Documentation

Additional Information about Noisy Datasets, Parameters and Reproduction can be found in [docs](docs)


## Citation

If you use this repository or reproduce results from the NEPAL paper, please cite:

(TBD)


## Contact

For questions or clarifications regarding the implementation or replication of experiments, please refer to the code repository or contact the paper authors.

## License
This code is licensed under [GPLv3](LICENSE.txt)