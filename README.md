# CSP5

Code for training CSP5 models, to accompany the paper "CSP5: Large-scale Neural Chemical Shift Prediction from 2.5 Million Experimental NMR Spectra".

This repo contains code for training the models. If you're interested in obtaining predictions using the models, we recommend using the accompanying PyPI package, `csp5`. This can be installed straightforwardly with `uv pip install csp5` in a suitable virtual environment. Package page and docs: [https://pypi.org/project/csp5/](https://pypi.org/project/csp5/).

## Environment setup

```bash
uv venv .venv --prompt .venv
UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Zenodo data
The accompanying Zenodo record ([https://zenodo.org/records/19486118](https://zenodo.org/records/19486118)) contains model files and the version of the NMRexp dataset used in this work. To download the data and extract it appropriately, run:

```bash
./scripts/download_zenodo_data.sh
```

## Build shards for training

Build both 13C and 1H ensemble shards directly from the Zenodo parquet:

```bash
./scripts/build_zenodo_ensemble_shards.sh --targets 13C,1H
```
This will do a conformational search for each structure with rdkit to obtain an appropriate conformational ensemble, and set up the data so that it can be used to train the models.

## Training

First train a model on the assigned data only (command for 13C shown):

```bash
python src/cascade_nmr/NMRexp_PaiNN/train_assigned.py \
  --target 13C \
  --entries-path zenodo_csp5_upload/data/assigned/Exp22K_13C_entries.pkl \
  --splits-path zenodo_csp5_upload/data/splits/CSP5-13C-scaffold-doi_split.json \
  --output-dir results/train_assigned_13c
```

Then train a joint model, on both the assigned and unassigned data, initialized from the assigned model:

```bash
python src/cascade_nmr/NMRexp_PaiNN/train_joint.py \
  --target 13C \
  --output-dir results/train_joint_13c \
  --ensemble-shards-dir data/cascade_nmrexp_13c_ensembles_scaffold_doi \
  --exp22k-entries-path zenodo_csp5_upload/data/assigned/Exp22K_13C_entries.pkl \
  --exp22k-splits-path zenodo_csp5_upload/data/splits/CSP5-13C-scaffold-doi_split.json \
  --init-from results/train_assigned_13c/best_model.pt
```

For 1H, set `--target 1H`, use matching 1H entries/splits/shard paths, and initialize from `results/train_assigned_1h/best_model.pt`.
