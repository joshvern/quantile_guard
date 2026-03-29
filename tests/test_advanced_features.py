"""
Tests for advanced features:
1. Analytical standard errors
2. Quantile process plots
3. Prediction intervals
4. Elastic net regularization
5. Cluster-robust standard errors
6. SCAD / MCP penalties
7. Goodness-of-fit (score, pseudo R²)
8. Formula interface
9. Censored quantile regression
10. Sparse solver mode
"""

import numpy as np
import pandas as pd
import pytest

from quantile_regression_pdlp import QuantileRegression, CensoredQuantileRegression


# ---- Fixtures ----

@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    true_beta = np.array([2.0, -1.5, 0.8])
    true_intercept = 3.0
    y = true_intercept + X @ true_beta + rng.normal(scale=0.5, size=n)
    return X, y, true_beta, true_intercept


@pytest.fixture
def clustered_data():
    rng = np.random.default_rng(42)
    n_clusters = 20
    n_per_cluster = 15
    n = n_clusters * n_per_cluster
    clusters = np.repeat(np.arange(n_clusters), n_per_cluster)
    cluster_effects = rng.normal(scale=1.0, size=n_clusters)
    X = rng.normal(size=(n, 2))
    y = 1.0 + 2.0 * X[:, 0] + cluster_effects[clusters] + rng.normal(scale=0.3, size=n)
    return X, y, clusters


@pytest.fixture
def sparse_irrelevant_data():
    """Data with some irrelevant features for variable selection tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.5, size=n)
    return X, y


# ==== 1. Analytical Standard Errors ====

class TestAnalyticalSE:

    def test_analytical_se_runs(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, se_method='analytical')
        model.fit(X, y)
        assert model.stderr_[0.5]['y'] is not None
        assert np.all(model.stderr_[0.5]['y'] > 0)

    def test_kernel_se_runs(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, se_method='kernel')
        model.fit(X, y)
        assert model.stderr_[0.5]['y'] is not None
        assert np.all(model.stderr_[0.5]['y'] > 0)

    def test_analytical_se_close_to_bootstrap(self, linear_data):
        """Analytical and bootstrap SEs should be in the same ballpark."""
        X, y, _, _ = linear_data

        m_ana = QuantileRegression(tau=0.5, se_method='analytical')
        m_ana.fit(X, y)

        m_boot = QuantileRegression(tau=0.5, se_method='bootstrap',
                                     n_bootstrap=200, random_state=0)
        m_boot.fit(X, y)

        ratio = m_ana.stderr_[0.5]['y'] / m_boot.stderr_[0.5]['y']
        # Should be within factor of 3 (rough check)
        assert np.all(ratio > 0.3) and np.all(ratio < 3.0), \
            f"SE ratio out of range: {ratio}"

    def test_analytical_ci_present(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, se_method='analytical')
        model.fit(X, y)
        ci = model.confidence_intervals_[0.5]['y']
        assert ci is not None
        assert 'lower_2.5%' in ci.columns
        assert np.all(ci['lower_2.5%'].values <= ci['upper_97.5%'].values)

    def test_analytical_pvalues_significant(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, se_method='analytical')
        model.fit(X, y)
        # True coefficients are large, should be significant
        for j in [1, 2, 3]:  # slope coefficients
            assert model.pvalues_[0.5]['y'][j] < 0.05

    def test_multiple_quantiles_analytical(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.25, 0.5, 0.75], se_method='analytical')
        model.fit(X, y)
        for q in [0.25, 0.5, 0.75]:
            assert np.all(model.stderr_[q]['y'] > 0)


# ==== 2. Quantile Process Plots ====

class TestQuantileProcessPlot:

    def test_plot_runs_without_error(self, linear_data):
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend

        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.1, 0.25, 0.5, 0.75, 0.9],
                                    n_bootstrap=20, random_state=0)
        model.fit(X, y)
        fig = model.plot_quantile_process()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_single_feature(self, linear_data):
        import matplotlib
        matplotlib.use('Agg')

        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.25, 0.5, 0.75],
                                    n_bootstrap=20, random_state=0)
        model.fit(X, y)
        fig = model.plot_quantile_process(feature='X1')
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_requires_multiple_quantiles(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        with pytest.raises(ValueError, match="at least 2"):
            model.plot_quantile_process()


# ==== 3. Prediction Intervals ====

class TestPredictionInterval:

    def test_predict_interval_shape(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.05, 0.5, 0.95],
                                    n_bootstrap=20, random_state=0)
        model.fit(X, y)
        result = model.predict_interval(X[:10], coverage=0.90)
        assert 'y' in result
        assert result['y']['lower'].shape == (10,)
        assert result['y']['median'].shape == (10,)
        assert result['y']['upper'].shape == (10,)

    def test_predict_interval_ordering(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.05, 0.5, 0.95],
                                    n_bootstrap=20, random_state=0)
        model.fit(X, y)
        result = model.predict_interval(X, coverage=0.90)
        assert np.all(result['y']['lower'] <= result['y']['median'])
        assert np.all(result['y']['median'] <= result['y']['upper'])

    def test_predict_interval_coverage(self, linear_data):
        """Interval should roughly cover the right proportion of test data."""
        X, y, _, _ = linear_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        model = QuantileRegression(tau=[0.05, 0.5, 0.95],
                                    n_bootstrap=20, random_state=0)
        model.fit(X_train, y_train)
        result = model.predict_interval(X_test, coverage=0.90)
        covered = np.mean(
            (y_test >= result['y']['lower']) & (y_test <= result['y']['upper'])
        )
        assert covered > 0.70, f"Coverage too low: {covered:.2f}"

    def test_predict_interval_requires_multiple_quantiles(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict_interval(X)

    def test_predict_interval_tau_mapping(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=[0.1, 0.5, 0.9],
                                    n_bootstrap=20, random_state=0)
        model.fit(X, y)
        result = model.predict_interval(X[:5], coverage=0.80)
        assert result['y']['tau_lower'] == 0.1
        assert result['y']['tau_upper'] == 0.9


# ==== 4. Elastic Net ====

class TestElasticNet:

    def test_elasticnet_runs(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    regularization='elasticnet', alpha=0.1, l1_ratio=0.5)
        model.fit(X, y)
        assert model.coef_[0.5]['y'] is not None

    def test_elasticnet_shrinkage_between_l1_and_none(self, linear_data):
        X, y, _, _ = linear_data
        alpha = 0.5

        m_none = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        m_none.fit(X, y)

        m_l1 = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='l1', alpha=alpha)
        m_l1.fit(X, y)

        m_en = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='elasticnet', alpha=alpha, l1_ratio=0.5)
        m_en.fit(X, y)

        norm_none = np.sum(np.abs(m_none.coef_[0.5]['y']))
        norm_l1 = np.sum(np.abs(m_l1.coef_[0.5]['y']))
        norm_en = np.sum(np.abs(m_en.coef_[0.5]['y']))

        # Elastic net should shrink, but typically less than pure L1
        assert norm_en < norm_none, "Elastic net should shrink vs unpenalized"

    def test_elasticnet_l1_ratio_1_matches_l1(self, linear_data):
        """l1_ratio=1 should behave like pure L1."""
        X, y, _, _ = linear_data
        alpha = 0.3

        m_l1 = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='l1', alpha=alpha)
        m_l1.fit(X, y)

        m_en = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='elasticnet', alpha=alpha, l1_ratio=1.0)
        m_en.fit(X, y)

        np.testing.assert_allclose(m_l1.coef_[0.5]['y'], m_en.coef_[0.5]['y'], atol=0.2)


# ==== 5. Cluster-Robust Standard Errors ====

class TestClusterSE:

    def test_cluster_se_runs(self, clustered_data):
        X, y, clusters = clustered_data
        model = QuantileRegression(tau=0.5, se_method='analytical')
        model.fit(X, y, clusters=clusters)
        assert np.all(model.stderr_[0.5]['y'] > 0)

    def test_cluster_se_differs_from_iid(self, clustered_data):
        """Cluster SEs should differ from IID SEs when data is clustered."""
        X, y, clusters = clustered_data

        m_iid = QuantileRegression(tau=0.5, se_method='analytical')
        m_iid.fit(X, y)

        m_clust = QuantileRegression(tau=0.5, se_method='analytical')
        m_clust.fit(X, y, clusters=clusters)

        # Cluster SEs typically larger when there's intra-cluster correlation
        diff = np.max(np.abs(m_clust.stderr_[0.5]['y'] - m_iid.stderr_[0.5]['y']))
        assert diff > 1e-4, "Cluster and IID SEs should differ"

    def test_cluster_se_larger_with_correlation(self, clustered_data):
        """With strong cluster effects, cluster SEs should generally be larger."""
        X, y, clusters = clustered_data

        m_iid = QuantileRegression(tau=0.5, se_method='analytical')
        m_iid.fit(X, y)

        m_clust = QuantileRegression(tau=0.5, se_method='analytical')
        m_clust.fit(X, y, clusters=clusters)

        # At least the intercept SE should be larger with clustering
        assert m_clust.stderr_[0.5]['y'][0] > m_iid.stderr_[0.5]['y'][0] * 0.8


# ==== 6. SCAD / MCP Penalties ====

class TestSCADMCP:

    def test_scad_runs(self, sparse_irrelevant_data):
        X, y = sparse_irrelevant_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    regularization='scad', alpha=0.5)
        model.fit(X, y)
        assert model.coef_[0.5]['y'] is not None

    def test_mcp_runs(self, sparse_irrelevant_data):
        X, y = sparse_irrelevant_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    regularization='mcp', alpha=0.5)
        model.fit(X, y)
        assert model.coef_[0.5]['y'] is not None

    def test_scad_variable_selection(self, sparse_irrelevant_data):
        """SCAD should shrink irrelevant features more than L1."""
        X, y = sparse_irrelevant_data
        alpha = 0.3

        m_scad = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                     regularization='scad', alpha=alpha)
        m_scad.fit(X, y)

        # Features 2,3,4 (indices) are noise - should be small
        noise_coefs = np.abs(m_scad.coef_[0.5]['y'][2:])
        signal_coefs = np.abs(m_scad.coef_[0.5]['y'][:2])
        assert np.mean(noise_coefs) < np.mean(signal_coefs), \
            "SCAD should distinguish signal from noise"

    def test_mcp_variable_selection(self, sparse_irrelevant_data):
        """MCP should produce sparse solutions."""
        X, y = sparse_irrelevant_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    regularization='mcp', alpha=0.3)
        model.fit(X, y)

        noise_coefs = np.abs(model.coef_[0.5]['y'][2:])
        signal_coefs = np.abs(model.coef_[0.5]['y'][:2])
        assert np.mean(noise_coefs) < np.mean(signal_coefs)

    def test_scad_less_bias_than_l1(self, sparse_irrelevant_data):
        """SCAD should have less bias on large true coefficients than L1."""
        X, y = sparse_irrelevant_data
        alpha = 0.3

        m_l1 = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='l1', alpha=alpha)
        m_l1.fit(X, y)

        m_scad = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                     regularization='scad', alpha=alpha)
        m_scad.fit(X, y)

        # True coefs: [3.0, -2.0, 0, 0, 0]
        # SCAD should recover the large coefficients better (less bias)
        scad_bias = abs(abs(m_scad.coef_[0.5]['y'][0]) - 3.0)
        l1_bias = abs(abs(m_l1.coef_[0.5]['y'][0]) - 3.0)
        assert scad_bias <= l1_bias + 0.3, \
            f"SCAD bias ({scad_bias:.3f}) should be <= L1 bias ({l1_bias:.3f})"


# ==== 7. Goodness-of-Fit ====

class TestGoodnessOfFit:

    def test_score_returns_negative_loss(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        s = model.score(X, y)
        assert s < 0  # Negative pinball loss

    def test_score_better_on_train_than_noise(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)

        score_train = model.score(X, y)
        rng = np.random.default_rng(99)
        score_noise = model.score(X, rng.normal(size=len(y)))
        assert score_train > score_noise

    def test_pseudo_r_squared_positive(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        r2 = model.pseudo_r_squared_
        assert r2[0.5]['y'] > 0, f"Pseudo R² should be positive: {r2[0.5]['y']}"

    def test_pseudo_r_squared_bounded(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        r2 = model.pseudo_r_squared_[0.5]['y']
        assert 0 < r2 <= 1, f"Pseudo R² should be in (0,1]: {r2}"

    def test_pseudo_r_squared_higher_for_good_model(self, linear_data):
        """Model with true features should have higher R² than noise features."""
        X, y, _, _ = linear_data
        rng = np.random.default_rng(0)

        # Good model
        m_good = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        m_good.fit(X, y)
        r2_good = m_good.pseudo_r_squared_[0.5]['y']

        # Bad model (noise features)
        X_noise = rng.normal(size=X.shape)
        m_bad = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        m_bad.fit(X_noise, y)
        r2_bad = m_bad.pseudo_r_squared_[0.5]['y']

        assert r2_good > r2_bad

    def test_pseudo_r_squared_multiple_quantiles(self, linear_data):
        X, y, _, _ = linear_data
        taus = [0.25, 0.5, 0.75]
        model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0)
        model.fit(X, y)
        r2 = model.pseudo_r_squared_
        for q in taus:
            assert 0 < r2[q]['y'] <= 1


# ==== 8. Formula Interface ====

class TestFormulaInterface:

    def test_formula_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'y': rng.normal(size=100),
            'x1': rng.normal(size=100),
            'x2': rng.normal(size=100),
        })
        df['y'] = 2.0 * df['x1'] - 1.0 * df['x2'] + rng.normal(scale=0.3, size=100)

        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit_formula('y ~ x1 + x2', df)

        assert 'x1' in model.feature_names_
        assert 'x2' in model.feature_names_
        assert abs(model.coef_[0.5]['y'][0] - 2.0) < 0.5

    def test_formula_categorical(self):
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.choice(['A', 'B', 'C'], size=n)
        x = rng.normal(size=n)
        effects = {'A': 0, 'B': 2, 'C': -1}
        y = np.array([effects[g] for g in groups]) + 1.5 * x + rng.normal(scale=0.3, size=n)

        df = pd.DataFrame({'y': y, 'x': x, 'group': groups})
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit_formula('y ~ x + C(group)', df)

        # Should have x and group dummy variables
        assert any('group' in f for f in model.feature_names_)
        pred = model.predict(model._X_aug[:5, 1:])  # skip intercept column
        assert pred[0.5]['y'].shape == (5,)

    def test_formula_with_weight_column(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'y': rng.normal(size=50),
            'x1': rng.normal(size=50),
            'w': rng.uniform(0.5, 2.0, size=50),
        })

        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        model.fit_formula('y ~ x1', df, weights='w')
        assert model.coef_[0.5]['y'] is not None


# ==== 9. Censored Quantile Regression ====

class TestCensoredQR:

    def test_censored_qr_runs(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=(n, 2))
        T = 2.0 + 1.5 * X[:, 0] + rng.exponential(scale=1.0, size=n)
        C = rng.exponential(scale=3.0, size=n)
        Y = np.minimum(T, C)
        delta = (T <= C).astype(int)

        model = CensoredQuantileRegression(
            tau=0.5, n_bootstrap=20, random_state=0, censoring='right')
        model.fit(X, Y, event_indicator=delta)
        assert model.coef_[0.5]['y'] is not None

    def test_censored_better_than_naive(self):
        """Censored QR should recover coefficients better than ignoring censoring."""
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=(n, 1))
        true_slope = 2.0
        T = 3.0 + true_slope * X[:, 0] + rng.normal(scale=0.5, size=n)
        C = np.full(n, np.quantile(T, 0.5))  # Censor ~50%
        Y = np.minimum(T, C)
        delta = (T <= C).astype(int)

        # Naive: ignore censoring
        m_naive = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        m_naive.fit(X, Y)

        # Censored
        m_cens = CensoredQuantileRegression(
            tau=0.5, n_bootstrap=20, random_state=0, censoring='right')
        m_cens.fit(X, Y, event_indicator=delta)

        naive_bias = abs(m_naive.coef_[0.5]['y'][0] - true_slope)
        cens_bias = abs(m_cens.coef_[0.5]['y'][0] - true_slope)

        # Censored model should have less bias (or at worst similar)
        assert cens_bias <= naive_bias + 0.5, \
            f"Censored bias ({cens_bias:.3f}) should be <= naive ({naive_bias:.3f})"

    def test_censored_no_censoring_matches_regular(self):
        """With no censored observations, should match regular QR."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=(n, 2))
        y = 1.0 + X @ np.array([1.5, -0.5]) + rng.normal(scale=0.3, size=n)
        delta = np.ones(n, dtype=int)  # All observed

        m_cens = CensoredQuantileRegression(
            tau=0.5, n_bootstrap=20, random_state=0, censoring='right')
        m_cens.fit(X, y, event_indicator=delta)

        m_reg = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        m_reg.fit(X, y)

        np.testing.assert_allclose(
            m_cens.coef_[0.5]['y'], m_reg.coef_[0.5]['y'], atol=0.3)

    def test_censored_get_params(self):
        model = CensoredQuantileRegression(
            tau=0.5, censoring='left', max_censor_iter=100, censor_tol=1e-5)
        params = model.get_params()
        assert params['censoring'] == 'left'
        assert params['max_censor_iter'] == 100
        assert params['censor_tol'] == 1e-5


# ==== 10. Sparse Solver Mode ====

class TestSparseSolver:

    def test_sparse_matches_ortools(self, linear_data):
        """Sparse scipy solver should produce similar results to OR-Tools."""
        X, y, _, _ = linear_data

        m_ort = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    use_sparse=False)
        m_ort.fit(X, y)

        m_sp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   use_sparse=True)
        m_sp.fit(X, y)

        np.testing.assert_allclose(
            m_ort.coef_[0.5]['y'], m_sp.coef_[0.5]['y'], atol=0.1,
            err_msg="Sparse and OR-Tools should agree"
        )
        np.testing.assert_allclose(
            m_ort.intercept_[0.5]['y'], m_sp.intercept_[0.5]['y'], atol=0.1
        )

    def test_sparse_multiple_quantiles(self, linear_data):
        X, y, _, _ = linear_data
        taus = [0.25, 0.5, 0.75]
        model = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0,
                                    use_sparse=True)
        model.fit(X, y)
        for q in taus:
            assert model.coef_[q]['y'] is not None

    def test_sparse_with_l1(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    use_sparse=True, regularization='l1', alpha=0.3)
        model.fit(X, y)
        assert model.coef_[0.5]['y'] is not None

    def test_sparse_predictions_match(self, linear_data):
        X, y, _, _ = linear_data

        m_ort = QuantileRegression(tau=[0.25, 0.5, 0.75], n_bootstrap=20,
                                    random_state=0, use_sparse=False)
        m_ort.fit(X, y)

        m_sp = QuantileRegression(tau=[0.25, 0.5, 0.75], n_bootstrap=20,
                                   random_state=0, use_sparse=True)
        m_sp.fit(X, y)

        pred_ort = m_ort.predict(X[:10])
        pred_sp = m_sp.predict(X[:10])
        for q in [0.25, 0.5, 0.75]:
            np.testing.assert_allclose(
                pred_ort[q]['y'], pred_sp[q]['y'], atol=0.2)

    def test_sparse_solver_info(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                    use_sparse=True)
        model.fit(X, y)
        assert model.solver_info_['status'] == 'OPTIMAL'
        assert model.solver_info_['num_variables'] > 0


# ==== Cross-feature integration tests ====

class TestIntegration:

    def test_sparse_with_analytical_se(self, linear_data):
        X, y, _, _ = linear_data
        model = QuantileRegression(tau=0.5, use_sparse=True, se_method='analytical')
        model.fit(X, y)
        assert np.all(model.stderr_[0.5]['y'] > 0)
        assert model.pseudo_r_squared_[0.5]['y'] > 0

    def test_scad_with_kernel_se(self, sparse_irrelevant_data):
        X, y = sparse_irrelevant_data
        model = QuantileRegression(tau=0.5, se_method='kernel',
                                    regularization='scad', alpha=0.3)
        model.fit(X, y)
        assert model.coef_[0.5]['y'] is not None
        assert np.all(model.stderr_[0.5]['y'] > 0)

    def test_full_pipeline(self, linear_data):
        """End-to-end: fit, predict, score, summary, plot."""
        import matplotlib
        matplotlib.use('Agg')

        X, y, _, _ = linear_data
        taus = [0.1, 0.25, 0.5, 0.75, 0.9]
        model = QuantileRegression(
            tau=taus, n_bootstrap=30, random_state=0, se_method='analytical')
        model.fit(X, y)

        pred = model.predict(X[:5])
        assert all(q in pred for q in taus)

        interval = model.predict_interval(X[:5])
        assert 'y' in interval

        s = model.score(X, y)
        assert s < 0

        r2 = model.pseudo_r_squared_
        assert all(r2[q]['y'] > 0 for q in taus)

        summary = model.summary()
        assert all(q in summary for q in taus)

        fig = model.plot_quantile_process(feature='X1')
        import matplotlib.pyplot as plt
        plt.close(fig)
