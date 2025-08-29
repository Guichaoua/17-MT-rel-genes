# 17 microtubule-related genes — code for Chapter 3

This repository accompanies Chapter 3 of my PhD thesis. It contains the code to:

1. Build a curated interaction network centered on the 17 microtubule-related (MT-rel) genes;

2. Reproduce the pCR prediction experiments (taxane-based chemotherapy; out-of-fold ROC–AUC) on public breast-cancer microarray cohorts.

## Setup —  Conda (`environment.yml`) 

You can reproduce the environment with Conda

```bash
# create the env
conda env create -f environment.yml
conda activate mt-rel-genes

# (optional) make it selectable in Jupyter
python -m ipykernel install --user --name mt-rel-genes --display-name "Python (mt-rel-genes)"
```