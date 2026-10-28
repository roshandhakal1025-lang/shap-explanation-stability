import numpy as np
import pytest
from src.experiment import dataset, model_for, pairwise_spearman, run

def test_data_split():
    Xtrain, Xtest, ytrain, ytest = dataset(n_samples=120)
    assert Xtrain.shape[1] == 14
    assert len(Xtrain) + len(Xtest) == 120
    assert len(ytrain) == len(Xtrain)
    assert len(ytest) == len(Xtest)

def test_pairwise_correlations():
    vectors = [[1, 2, 3], [1, 2, 3], [3, 2, 1]]
    values = pairwise_spearman(vectors)
    assert len(values) == 3
    assert np.allclose(sorted(values), [-1, -1, 1])

def test_invalid_model():
    with pytest.raises(ValueError):
        model_for("unknown", 42)

def test_reproducible_small_run():
    first = run(seed=7, bootstraps=2)
    second = run(seed=7, bootstraps=2)
    assert first == second
    assert first["importance_method"].startswith("permutation")
