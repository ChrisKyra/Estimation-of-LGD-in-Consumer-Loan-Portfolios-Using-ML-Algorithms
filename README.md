# Estimation of Loss Given Default in Consumer Loan Portfolios Using Machine Learning Algorithms

This repository contains the full implementation and documentation of my master's thesis titled **"Estimation of Loss Given Default in Consumer Loan Portfolios Using Machine Learning Algorithms"**.
The project was completed as part of the MSc in ECE Auth program.

## 📄 Thesis Overview

The thesis focuses on the prediction of **Loss Given Default (LGD)** — a critical metric in credit risk modeling — by forecasting two key components:

- **Cure Rate**: The probability that a loan in default returns to performing status.
- **Recovery Rate**: The proportion of the loan recovered after default.

The two are combined with the formula used by ING (thesis, eq. 1):

```
LGD = (1 − cure rate) × (1 − recovery rate) + costs / EAD
```

## 🗂️ Repository Structure

```
├── Christos Kyratsous.pdf         thesis
├── RESULTS.xlsx                   results reported in the thesis
├── Δεδομένα/                      data
│   ├── Updated.xlsx               raw loan-level data
│   └── Dataset_with_rec_rate.xlsx processed data used for the thesis results
├── Κώδικας/                       original thesis notebooks (TensorFlow)
├── notebooks/                     upgraded pipeline: PyTorch on Apple Silicon (MPS)
│   ├── 01_build_dataset.ipynb     features, encoding, train/valid/test split, IV & AUC
│   ├── 02_cure_models.ipynb       cure probability: LR, RF, XGBoost, neural networks, ensemble
│   ├── 03_recovery_models.ipynb   recovery rate: LR, fractional logit, (inflated) beta regression, RF, XGBoost, NN, stacking
│   ├── 04_lgd_estimation.ipynb    combines both into LGD, loan level and portfolio level
│   └── lgd/                       shared code (data, metrics, plots, PyTorch models, Optuna tuning)
├── environment.yml / requirements.txt
└── run_all.sh                     runs notebooks 01–04 in order
```

## ⚡ PyTorch on Apple Silicon

The neural networks of the thesis were rewritten from TensorFlow/Keras to **PyTorch** and train on the Apple GPU
through PyTorch's **MPS** (Metal Performance Shaders) backend (`notebooks/lgd/torch_models.py`):

- The device is chosen automatically (`mps` when available, otherwise `cpu`; override with `LGD_DEVICE`).
- The whole training set is copied to the GPU once and mini-batches are sliced on the GPU, avoiding per-batch
  CPU → GPU copies.
- Early stopping on the validation loss restores the best weights.
- On the GPU: the cure networks, the recovery network, fractional logit, beta and zero-one inflated beta regression,
  and the network of the LightGBM → NN stacking model.
- Random Forest, XGBoost and LightGBM run on the CPU cores (they have no Apple GPU support).

## 🧠 Methodology (upgraded pipeline)

- **Feature availability**: by default only features known at the moment of default are used
  (`at_default`); the thesis feature set (`thesis`) is kept for comparison.
- **No data leakage**: one stratified 70/10/20 split for all notebooks; scaling, correlation analysis and IV
  are computed on the training rows; hyperparameters are tuned on CV / the validation set; the test set is only
  used for the final report.
- **Encoding** without the dummy-variable trap (`country` is implied by `city`; one reference category per variable).
- **Cure models** evaluated with accuracy, ROC-AUC, Gini, Brier score, log-loss and calibration plots.
- **Recovery models** trained on non-cured loans (the only ones where the recovery rate enters the LGD), including
  the LGD-literature baselines fractional logit, beta regression and zero-one inflated beta regression, and a
  two-part (0/1-inflated) XGBoost; the recovery model is chosen by repeated cross-validation.
- **Hyperparameter tuning** with Optuna (TPE sampler, pruning for neural networks); XGBoost boosting rounds chosen by
  cross-validated early stopping.
- **Ensembles** of the three best models.
- **Stacking** of LightGBM and a neural network with out-of-fold predictions.
- **Uncertainty**: 95% bootstrap confidence intervals on the test metrics.
- **Interpretability**: coefficients of the linear models and SHAP values of XGBoost.
- **LGD**: loan-level accuracy, EAD-weighted portfolio LGD and decile calibration.

## 🛠️ Technologies Used

- Python 3.12, PyTorch (MPS backend on Apple Silicon), scikit-learn, XGBoost, LightGBM, Optuna, SHAP
- Pandas, NumPy, Matplotlib, Seaborn, Jupyter

## ▶️ How to Run (macOS, Apple Silicon)

```bash
conda env create -f environment.yml
conda activate lgd
python -m ipykernel install --user --name lgd --display-name "Python (lgd)"
```

Then either open the notebooks in Jupyter / VS Code with the **Python (lgd)** kernel and run them in order
(01 → 04), or run everything from the terminal:

```bash
./run_all.sh --quick   # smoke test: a few minutes, meaningless numbers
./run_all.sh           # full run
```

Results are written to `outputs/` (tables and predictions) and `outputs/figures/`.

| Environment variable | Effect |
|---|---|
| `LGD_QUICK=1` | 3 trials / 3 epochs per model, to check that everything runs |
| `LGD_DEVICE=cpu` | train the PyTorch models on the CPU instead of the GPU |
| `LGD_OUTPUT_DIR=…` | write outputs somewhere other than `outputs/` |
