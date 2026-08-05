# Reproduce our Experiments
These are the steps required to reproduce the results we reported in our paper.

**Note:** Encoding generation, data splitting, hyperparameter search, model training,
embedding, and alignment use `GLOBAL_CONFIG.Seed` (42 by default). The default single-trial
scheduler and deterministic PyTorch setting make repeated runs on the same software and
hardware stack as stable as practical. Floating-point results can still differ across
operating systems, accelerators, and driver versions. Parallel asynchronous HPO can also
change the order in which Optuna observes trials; use `--parallel-trials 1` when exact
repeatability matters more than runtime.

**Another Note:** Re-Running all experiments will take a considerable amount of time. Depending on your
system specification you might face runtimes in excess of a week.
This is due to the large number of parameter combinations.
___
### System Details

The experiments were run on a virtual machine with the following specification:

- Ubuntu 24.04 LTS
- 20 cores of an AMD EPYC 9254
- NVIDIA GeForce RTX 3090 Ti, 24 GB VRAM
- 176 GB of RAM
- 3 TB HDD space

### Docker Environment

The Docker image is based on `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`, matching the pinned `torch==2.2.0` and `torchvision==0.17.0` versions in `requirements.txt`.
The image build verifies those versions after dependency installation.

Clone the repository and its pinned submodule revision, then run the configuration and input preflight:

```bash
git clone --recurse-submodules https://github.com/marcelmildenberger/neural-pattern-learning-attack.git
cd neural-pattern-learning-attack
docker build -t nepal .
docker run --rm -v "$(pwd):/usr/app" nepal python3 main.py --config nepal_config.json --validate-only
```

For a native setup, use Python 3.10 and install `requirements.txt` on Linux or
`requirements_macOS.txt` on macOS. All declared dependency versions are exact pins.

___
### Obtain Datasets
Make sure that you have all required datasets in the  `./data` directory.
The code expects the following files to be present:

```
fakename_1k.tsv     fakename_2k.tsv     fakename_5k.tsv     fakename_10k.tsv 
fakename_20k.tsv    fakename_50k.tsv    euro_person.tsv     titanic_full.tsv
```

Remember to [prepare](../README.md) the dataset so it fits the correct file format.
**Note:** To run the attack on a synthetic dataset, you need to provide an encoded version of the dataset for BF, TMH and TSH where the encoding is provided before the uid column.
Clean encoded FakeName files are not stored in the repository by default. Regenerate them with:

``python3 encode_datasets.py --source-dir data/datasets --encoders bf tmh tsh --seed 42``

Use `--encoders bfd` as well if you want diffused Bloom filter files.

The Euro Person dataset needs to be downloaded and prepared accordingly using the dataset provided here: [Download](https://wayback.archive-it.org/12090/20231229131836/http://ec.europa.eu/eurostat/cros/system/files/Transfer%20to%20Istat.zip)

For smaller repository-friendly dataset bundles, see [Dataset Publication Strategy](dataset_publication.md).


___
### Run the Benchmarks
To reproduce the results we reported in our paper, you may simply run

``python3 experiment_setup.py``

**Note:** By default, synthetically created data splits will be used. To enable the scenario GMA-NEPAL, the Graph Matching Attack needs to be enabled (see [parameters.md](parameters.md)).

To inspect the full matrix without running it:

``python3 experiment_setup.py --dry-run``

Validate the default single-run configuration and its input files before committing compute:

``python3 main.py --config nepal_config.json --validate-only``

To run a smaller custom matrix:

``python3 experiment_setup.py --datasets fakename_1k.tsv fakename_5k.tsv --encoders bf tsh --overlaps 0.2 0.8 --num-samples 25 --epochs 10 --seed 42 --parallel-trials 1``

Common options include `--train-size`, `--parallel-trials`, `--no-gpu`, `--max-runs`, and `--bf-diffusion`.

To run the same matrix on clean datasets instead of noisy encoded datasets:

``python3 experiment_setup.py --clean``

If encoded inputs are missing, the runner reports the exact missing files and the `encode_datasets.py` command needed to regenerate them.

To run GMA-NEPAL:

``python3 experiment_setup.py --graph-matching --drop-from Eve Both``

The CLI accepts encoder aliases (`bf`, `tmh`, `tsh`) and full encoder names (`BloomFilter`, `TabMinHash`, `TwoStepHash`).
___
### Reproduce Plots
Once the benchmark is complete, you can generate the result plots used in our paper.
Simply generate the plots by running

``python3 extract_nepal_results.py``

and then run the ``analysis.ipynb`` notebook
