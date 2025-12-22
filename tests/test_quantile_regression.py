import numpy as np
import pytest

from quantile_regression_pdlp import QuantileRegression


def _make_synthetic_regression(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([1.5, -2.0, 0.5])[:n_features]
    y = X @ beta + rng.normal(scale=0.2, size=n_samples)
    return X, y


def test_fit_and_predict_shape_single_quantile():
    X, y = _make_synthetic_regression()

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0, n_jobs=1)
    model.fit(X, y)

    y_pred = model.predict(X[:5])
    assert 0.5 in y_pred
    assert "y" in y_pred[0.5]
    assert y_pred[0.5]["y"].shape == (5,)


def test_multi_quantile_support_predict_shapes():
    X, y = _make_synthetic_regression(seed=1)

    taus = [0.25, 0.5, 0.75]
    model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0, n_jobs=1)
    model.fit(X, y)

    y_pred = model.predict(X[:7])

    assert set(y_pred.keys()) == set(taus)
    for q in taus:
        assert "y" in y_pred[q]
        assert y_pred[q]["y"].shape == (7,)


def test_multi_output_support_predict_shapes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))

    y1 = 1.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=50)
    y2 = -0.3 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(scale=0.1, size=50)
    Y = np.column_stack([y1, y2])

    model = QuantileRegression(tau=[0.4, 0.6], n_bootstrap=20, random_state=0, n_jobs=1)
    model.fit(X, Y)

    y_pred = model.predict(X[:4])
    for q in [0.4, 0.6]:
        assert set(y_pred[q].keys()) == {"y1", "y2"}
        assert y_pred[q]["y1"].shape == (4,)
        assert y_pred[q]["y2"].shape == (4,)
