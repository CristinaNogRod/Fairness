# FAIR Score guided Outlier Detection (FSGOD)

This repository contains the source code and results generated for the paper: [TODO](www.google.com)


## Setup

**Install dependencies**
```bash
python -m pip install -r requirements.txt
```

## Used Datasets

* **Adult**: https://archive.ics.uci.edu/ml/datasets/adult 
* **UCI Credit**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients 
* **Insurance**: https://www.kaggle.com/mirichoi0218/insurance
...

## Downloading and processing datasets

The `download_data.py` and `process_data.py` modules handle the downloading and processing of the datasets. You can manually
download and process them with the following commands (**run from the project's root directory**):

```
# from project's root folder
python src/data/download_data.py all  # Downloads all datasets. Specify an ID for downloading a particular one
python src/data/process_data.py all   # Builds all datasets. Specify an ID for crafting a particular one
```

The first command will download a raw copy of the datasets under `datasets/raw/[NAME].csv`, whereas the second one will store a processed 
copy under `datasets/proc/crafted_[NAME].csv`. All of the subsequent training/evaluation scripts will use the processed datasets, 
so, if you wish to do so, you can remove the raw versions.

## Running an experiment

Experiments are organized by .py scripts. Each one of them will run a particular experiment and save the results under 
the `outputs/` folder. You can run them **from the project's root directory** with:

```
python src/experiments/[experiment_name].py
```

