import numpy as np
import pandas as pd
import pytest

from quantile_guard import QuantileRegression


def _make_synthetic_regression(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([1.5, -2.0, 0.5])[:n_features]
    y = X @ beta + rng.normal(scale=0.2, size=n_samples)
    return X, y, beta


# ---- Shape tests (original) ----

def test_fit_and_predict_shape_single_quantile():
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0, n_jobs=1)
    model.fit(X, y)
    y_pred = model.predict(X[:5])
    assert 0.5 in y_pred
    assert "y" in y_pred[0.5]
    assert y_pred[0.5]["y"].shape == (5,)


def test_multi_quantile_support_predict_shapes():
    X, y, _ = _make_synthetic_regression(seed=1)
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


# ---- Coefficient correctness tests ----

def test_median_regression_recovers_true_coefficients():
    """Median regression (tau=0.5) should recover true slope and intercept."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 1))
    true_intercept = 3.0
    true_slope = 2.0
    y = true_intercept + true_slope * X[:, 0] + rng.normal(scale=0.3, size=n)

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    assert abs(model.intercept_[0.5]['y'] - true_intercept) < 0.3
    assert abs(model.coef_[0.5]['y'][0] - true_slope) < 0.3


def test_multivariate_coefficient_recovery():
    """Multi-feature regression should recover all true coefficients."""
    X, y, true_beta = _make_synthetic_regression(n_samples=200, seed=10)

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    recovered = model.coef_[0.5]['y']
    for j in range(len(true_beta)):
        assert abs(recovered[j] - true_beta[j]) < 0.3, \
            f"Feature {j}: expected {true_beta[j]}, got {recovered[j]}"


def test_quantile_ordering_in_coefficients():
    """Lower quantiles should have lower intercepts for symmetric noise."""
    rng = np.random.default_rng(99)
    n = 300
    X = rng.normal(size=(n, 1))
    y = 5.0 + 1.0 * X[:, 0] + rng.normal(scale=1.0, size=n)

    taus = [0.1, 0.5, 0.9]
    model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    intercepts = [model.intercept_[q]['y'] for q in taus]
    assert intercepts[0] < intercepts[1] < intercepts[2], \
        f"Intercepts should be ordered: {intercepts}"


# ---- L1 regularization tests ----

def test_l1_regularization_shrinks_coefficients():
    """L1 regularization should shrink coefficients toward zero."""
    X, y, _ = _make_synthetic_regression(n_samples=100, seed=5)

    model_none = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0, regularization='none')
    model_none.fit(X, y)

    model_l1 = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='l1', alpha=1.0)
    model_l1.fit(X, y)

    coef_none = np.abs(model_none.coef_[0.5]['y'])
    coef_l1 = np.abs(model_l1.coef_[0.5]['y'])

    assert np.sum(coef_l1) < np.sum(coef_none), \
        f"L1 should shrink: |coef_l1|={np.sum(coef_l1):.4f} vs |coef_none|={np.sum(coef_none):.4f}"


def test_strong_l1_near_zero_coefficients():
    """Very strong L1 penalty should drive coefficients close to zero."""
    X, y, _ = _make_synthetic_regression(n_samples=100, seed=5)

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                regularization='l1', alpha=100.0)
    model.fit(X, y)

    coef = model.coef_[0.5]['y']
    assert np.all(np.abs(coef) < 0.5), f"Strong L1 coefficients too large: {coef}"


# ---- Non-crossing tests ----

def test_non_crossing_constraints_at_training():
    """Predicted quantiles should not cross on training data."""
    rng = np.random.default_rng(7)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=n)

    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    y_pred = model.predict(X)
    for i in range(n):
        preds = [y_pred[q]['y'][i] for q in taus]
        for j in range(len(preds) - 1):
            assert preds[j] <= preds[j + 1] + 1e-6, \
                f"Crossing at obs {i}: tau={taus[j]}={preds[j]:.4f} > tau={taus[j+1]}={preds[j+1]:.4f}"


def test_non_crossing_enforced_at_prediction():
    """Non-crossing should be enforced at prediction time on new data."""
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(80, 2))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + rng.normal(scale=1.0, size=80)

    X_test = rng.normal(size=(50, 2)) * 3  # Extrapolation

    taus = [0.1, 0.5, 0.9]
    model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0,
                                enforce_non_crossing_predict=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    for i in range(50):
        preds = [y_pred[q]['y'][i] for q in taus]
        for j in range(len(preds) - 1):
            assert preds[j] <= preds[j + 1] + 1e-6, \
                f"Prediction crossing at obs {i}"


# ---- Weighted regression test ----

def test_weighted_regression_differs_from_unweighted():
    """Weighted regression should produce different coefficients than unweighted."""
    rng = np.random.default_rng(11)
    n = 100
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.5, size=n)

    # Up-weight observations where X[:,0] > 0
    weights = np.where(X[:, 0] > 0, 5.0, 1.0)

    model_uw = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model_uw.fit(X, y)

    model_w = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model_w.fit(X, y, weights=weights)

    diff = np.max(np.abs(model_w.coef_[0.5]['y'] - model_uw.coef_[0.5]['y']))
    assert diff > 1e-4, "Weighted and unweighted should differ"


# ---- Solver diagnostics tests ----

def test_solver_info_populated():
    """solver_info_ should be populated after fit."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    assert model.solver_info_ is not None
    assert model.solver_info_['status'] == 'OPTIMAL'
    assert model.solver_info_['wall_time_seconds'] >= 0
    assert model.solver_info_['num_variables'] > 0
    assert model.solver_info_['num_constraints'] > 0
    assert model.solver_info_['objective_value'] is not None


def test_glop_solver_backend():
    """GLOP backend should produce valid results."""
    X, y, _ = _make_synthetic_regression(n_samples=50, seed=3)
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                solver_backend='GLOP')
    model.fit(X, y)

    y_pred = model.predict(X[:5])
    assert y_pred[0.5]['y'].shape == (5,)
    assert model.solver_info_['status'] == 'OPTIMAL'


def test_solver_time_limit():
    """Solver should respect time limit parameter without error on small problems."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                solver_time_limit=60.0)
    model.fit(X, y)
    assert model.solver_info_['status'] == 'OPTIMAL'


def test_invalid_solver_backend():
    """Invalid solver backend should raise ValueError."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                solver_backend='NONEXISTENT')
    with pytest.raises(ValueError, match="not available"):
        model.fit(X, y)


# ---- Empirical bootstrap inference tests ----

def test_pvalues_significant_for_true_signal():
    """True non-zero coefficients should have small p-values."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 1))
    y = 5.0 + 3.0 * X[:, 0] + rng.normal(scale=0.3, size=n)

    model = QuantileRegression(tau=0.5, n_bootstrap=100, random_state=0)
    model.fit(X, y)

    # Slope coefficient p-value should be significant
    p_slope = model.pvalues_[0.5]['y'][1]  # index 1 = slope
    assert p_slope < 0.05, f"Slope p-value should be significant: {p_slope}"


def test_pvalues_non_significant_for_noise():
    """Pure noise features should have large p-values."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 2))
    y = 1.0 + rng.normal(scale=1.0, size=n)  # No relationship with X

    model = QuantileRegression(tau=0.5, n_bootstrap=100, random_state=0)
    model.fit(X, y)

    for j in [1, 2]:  # indices 1,2 = feature coefficients
        p = model.pvalues_[0.5]['y'][j]
        assert p > 0.05, f"Noise feature {j} p-value should be non-significant: {p}"


def test_confidence_intervals_present():
    """Confidence intervals should be populated after fit."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    ci = model.confidence_intervals_[0.5]['y']
    assert ci is not None
    assert 'lower_2.5%' in ci.columns
    assert 'upper_97.5%' in ci.columns
    assert len(ci) == 4  # intercept + 3 features
    # Lower should be < upper
    assert np.all(ci['lower_2.5%'].values <= ci['upper_97.5%'].values)


def test_confidence_intervals_cover_true_value():
    """95% CI should generally cover the true coefficient."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 1))
    true_slope = 2.0
    y = 1.0 + true_slope * X[:, 0] + rng.normal(scale=0.3, size=n)

    model = QuantileRegression(tau=0.5, n_bootstrap=200, random_state=0)
    model.fit(X, y)

    ci = model.confidence_intervals_[0.5]['y']
    slope_lower = ci.loc['X1', 'lower_2.5%']
    slope_upper = ci.loc['X1', 'upper_97.5%']
    assert slope_lower < true_slope < slope_upper, \
        f"CI [{slope_lower:.3f}, {slope_upper:.3f}] should cover true slope {true_slope}"


# ---- Summary test ----

def test_summary_includes_confidence_intervals():
    """Summary should include CI columns."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    summary = model.summary()
    df = summary[0.5]['y']
    assert 'CI 2.5%' in df.columns
    assert 'CI 97.5%' in df.columns
    assert 'Coefficient' in df.columns
    assert 'P>|t|' in df.columns


# ---- Input validation tests ----

def test_tau_validation_out_of_range():
    """tau outside (0,1) should raise ValueError."""
    with pytest.raises(ValueError):
        model = QuantileRegression(tau=1.5)
        model.fit(np.array([[1], [2]]), np.array([1, 2]))


def test_tau_validation_zero():
    """tau=0 should raise ValueError."""
    with pytest.raises(ValueError):
        model = QuantileRegression(tau=0.0)
        model.fit(np.array([[1], [2]]), np.array([1, 2]))


def test_weights_wrong_length():
    """Mismatched weights should raise ValueError."""
    X, y, _ = _make_synthetic_regression()
    model = QuantileRegression(tau=0.5, n_bootstrap=20)
    with pytest.raises(ValueError, match="same length"):
        model.fit(X, y, weights=np.ones(5))


def test_predict_before_fit():
    """Calling predict before fit should raise."""
    model = QuantileRegression(tau=0.5)
    with pytest.raises(Exception):
        model.predict(np.array([[1, 2, 3]]))


# ---- Pandas integration tests ----

def test_pandas_input_preserves_names():
    """Feature and output names from pandas should be preserved."""
    rng = np.random.default_rng(0)
    df_X = pd.DataFrame(rng.normal(size=(40, 2)), columns=['age', 'income'])
    df_y = pd.Series(rng.normal(size=40), name='price')

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(df_X, df_y)

    assert model.feature_names_ == ['age', 'income']
    assert model.output_names_ == ['price']

    summary = model.summary()
    assert 'age' in summary[0.5]['price'].index
    assert 'income' in summary[0.5]['price'].index


# ---- get_params / set_params tests ----

def test_get_params_includes_new_params():
    """get_params should include all new parameters."""
    model = QuantileRegression(solver_backend='GLOP', solver_tol=1e-6,
                                solver_time_limit=30.0, enforce_non_crossing_predict=False)
    params = model.get_params()
    assert params['solver_backend'] == 'GLOP'
    assert params['solver_tol'] == 1e-6
    assert params['solver_time_limit'] == 30.0
    assert params['enforce_non_crossing_predict'] is False


def test_set_params_roundtrip():
    """set_params should update parameters correctly."""
    model = QuantileRegression()
    model.set_params(solver_backend='GLOP', alpha=0.5)
    assert model.solver_backend == 'GLOP'
    assert model.alpha == 0.5


# ---- Single feature edge case ----

def test_single_feature():
    """Model should work with a single feature."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 1))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.2, size=50)

    model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
    model.fit(X, y)

    y_pred = model.predict(X[:3])
    assert y_pred[0.5]['y'].shape == (3,)
    assert abs(model.coef_[0.5]['y'][0] - 2.0) < 0.3


def test_sklearn_pipeline_compatibility():
    """Estimator should fit and predict inside a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _ = _make_synthetic_regression(n_samples=80, seed=12)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("qr", QuantileRegression(tau=[0.25, 0.5, 0.75], se_method="analytical")),
        ]
    )

    pipe.fit(X, y)
    pred = pipe.predict(X[:5])

    assert set(pred.keys()) == {0.25, 0.5, 0.75}
    assert pred[0.5]["y"].shape == (5,)
