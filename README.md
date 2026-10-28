# Stability of Feature Explanations for Intrusion Detection

Reproducible **starter research project** examining how stable feature-importance rankings are across bootstrap retraining and model families.

## What this implementation does
- Creates a **synthetic** classification dataset with a fixed seed.
- Fits Random Forest, Extra Trees, and HistGradientBoosting models on bootstrap samples.
- Uses permutation importance (not SHAP) and measures Spearman rank correlations.
- Writes structured results to `results/stability.json`.
- Includes automated tests.

**Research integrity:** This starter does not yet use UNSW-NB15 or SHAP and does not substantiate numerical findings currently described in a graduate resume. Do not describe its results as UNSW-NB15/SHAP research until those experiments have been performed and verified.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m src.experiment --bootstraps 8
python -m pytest -q
```

The generated `results/` folder is ignored by Git. Future work: integrate a properly sourced UNSW-NB15 dataset, SHAP TreeExplainer, attack-family holdouts, confidence intervals, and reproducible configuration.
