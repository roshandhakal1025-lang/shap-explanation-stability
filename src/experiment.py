"""Permutation-importance stability experiments on SYNTHETIC data.

This is NOT a SHAP or UNSW-NB15 benchmark. Do not report it as one.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

def dataset(seed=42, n_samples=900):
    X, y = make_classification(n_samples=n_samples, n_features=14, n_informative=8,
        n_redundant=3, random_state=seed)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

def model_for(name, seed):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    raise ValueError(f"Unknown model: {name}")

def pairwise_spearman(vectors):
    scores = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            value = float(spearmanr(vectors[i], vectors[j]).statistic)
            if np.isfinite(value):
                scores.append(value)
    return scores

def run(seed=42, bootstraps=8):
    if bootstraps < 2:
        raise ValueError("At least two bootstrap replicates required")
    Xtrain, Xtest, ytrain, ytest = dataset(seed)
    rng = np.random.default_rng(seed)
    names = ["random_forest", "extra_trees", "hist_gradient_boosting"]
    importance = {name: [] for name in names}
    for replicate in range(bootstraps):
        indices = rng.choice(len(Xtrain), size=len(Xtrain), replace=True)
        for name in names:
            model = model_for(name, seed + replicate)
            model.fit(Xtrain[indices], ytrain[indices])
            scores = permutation_importance(model, Xtest, ytest, scoring="recall",
                n_repeats=3, random_state=seed + replicate, n_jobs=1)
            importance[name].append(scores.importances_mean.tolist())
    within = {name: pairwise_spearman(vectors) for name, vectors in importance.items()}
    across = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            across[first + "_vs_" + second] = float(spearmanr(
                np.mean(importance[first], axis=0), np.mean(importance[second], axis=0)).statistic)
    return {
        "dataset": "synthetic sklearn classification (NOT UNSW-NB15)",
        "importance_method": "permutation importance (NOT SHAP)",
        "seed": seed,
        "bootstrap_replicates": bootstraps,
        "within_model_spearman_mean": {name: round(float(np.mean(scores)), 4) if scores else None for name, scores in within.items()},
        "between_model_spearman": {name: round(value, 4) for name, value in across.items()},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--output", default="results/stability.json")
    args = parser.parse_args()
    result = run(args.seed, args.bootstraps)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
