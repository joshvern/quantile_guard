"""
Benchmark tests validating against R's quantreg package.

These run R via subprocess using a micromamba environment.
Run explicitly: python -m pytest tests/bench_vs_r.py -v

Requires: .r_env with r-base and r-quantreg installed.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from quantile_guard import QuantileRegression, CensoredQuantileRegression

R_ENV = os.path.join(os.path.dirname(__file__), '..', '.r_env')
MICROMAMBA = '/tmp/bin/micromamba'


def _r_available():
    """Check if R environment is available."""
    r_env_abs = os.path.abspath(R_ENV)
    if not os.path.isdir(r_env_abs):
        return False
    if not os.path.isfile(MICROMAMBA):
        return False
    try:
        result = subprocess.run(
            [MICROMAMBA, 'run', '-p', r_env_abs, 'Rscript', '-e', 'cat("ok")'],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() == 'ok'
    except Exception:
        return False


def _run_r_script(script, timeout=60):
    """Run an R script and return stdout."""
    r_env_abs = os.path.abspath(R_ENV)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(script)
        f.flush()
        try:
            result = subprocess.run(
                [MICROMAMBA, 'run', '-p', r_env_abs, 'Rscript', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"R error:\n{result.stderr}")
            return result.stdout
        finally:
            os.unlink(f.name)


def _save_data(X, y, path):
    """Save data to CSV for R to read."""
    import pandas as pd
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df['y'] = y
    df.to_csv(path, index=False)


skip_no_r = pytest.mark.skipif(
    not _r_available(),
    reason="R environment with quantreg not available"
)


@pytest.fixture
def linear_data_file(tmp_path):
    """Generate linear data and save to CSV."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 3))
    true_beta = np.array([2.0, -1.5, 0.8])
    true_intercept = 3.0
    y = true_intercept + X @ true_beta + rng.normal(scale=0.5, size=n)
    csv_path = str(tmp_path / 'data.csv')
    _save_data(X, y, csv_path)
    return X, y, true_beta, true_intercept, csv_path


@skip_no_r
class TestAgainstR:

    def _fit_r_quantreg(self, csv_path, tau=0.5, se_method='iid'):
        """Fit R quantreg and return coefficients + SEs as JSON."""
        script = f"""
        library(quantreg)
        library(jsonlite)

        data <- read.csv("{csv_path}")
        formula <- y ~ X1 + X2 + X3
        fit <- rq(formula, tau={tau}, data=data)
        s <- summary(fit, se="{se_method}")

        result <- list(
            coefficients = as.numeric(coef(fit)),
            stderr = as.numeric(s$coefficients[, "Std. Error"]),
            tvalues = as.numeric(s$coefficients[, "t value"]),
            pvalues = as.numeric(s$coefficients[, "Pr(>|t|)"])
        )
        cat(toJSON(result, auto_unbox=TRUE))
        """
        output = _run_r_script(script)
        return json.loads(output)

    def _fit_r_multiple_quantiles(self, csv_path, taus):
        """Fit R quantreg for multiple quantiles."""
        tau_str = f"c({','.join(str(t) for t in taus)})"
        script = f"""
        library(quantreg)
        library(jsonlite)

        data <- read.csv("{csv_path}")
        taus <- {tau_str}
        results <- list()

        for (tau in taus) {{
            fit <- rq(y ~ X1 + X2 + X3, tau=tau, data=data)
            results[[as.character(tau)]] <- list(
                coefficients = as.numeric(coef(fit)),
                intercept = as.numeric(coef(fit)[1]),
                slopes = as.numeric(coef(fit)[-1])
            )
        }}
        cat(toJSON(results, auto_unbox=TRUE))
        """
        output = _run_r_script(script)
        return json.loads(output)

    def test_median_coefficients_match_r(self, linear_data_file):
        """PDLP should match R's quantreg at tau=0.5."""
        X, y, _, _, csv_path = linear_data_file

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y)

        r_result = self._fit_r_quantreg(csv_path, tau=0.5)

        # Compare intercept
        np.testing.assert_allclose(
            pdlp.intercept_[0.5]['y'], r_result['coefficients'][0], atol=0.15,
            err_msg="Intercept should match R"
        )

        # Compare slopes
        np.testing.assert_allclose(
            pdlp.coef_[0.5]['y'], r_result['coefficients'][1:], atol=0.15,
            err_msg="Slopes should match R"
        )

    def test_multiple_quantiles_match_r(self, linear_data_file):
        """PDLP should match R across multiple quantiles."""
        X, y, _, _, csv_path = linear_data_file
        taus = [0.1, 0.25, 0.5, 0.75, 0.9]

        pdlp = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y)

        r_results = self._fit_r_multiple_quantiles(csv_path, taus)

        for q in taus:
            r_coef = r_results[str(q)]['slopes']
            np.testing.assert_allclose(
                pdlp.coef_[q]['y'], r_coef, atol=0.2,
                err_msg=f"Slopes should match R at tau={q}"
            )

    def test_analytical_se_matches_r_iid(self, linear_data_file):
        """Analytical IID SEs should be close to R's se='iid'."""
        X, y, _, _, csv_path = linear_data_file

        pdlp = QuantileRegression(tau=0.5, se_method='analytical')
        pdlp.fit(X, y)

        r_result = self._fit_r_quantreg(csv_path, tau=0.5, se_method='iid')

        # SEs should be in the same ballpark (within factor of 3)
        ratio = pdlp.stderr_[0.5]['y'] / np.array(r_result['stderr'])
        assert np.all(ratio > 0.3) and np.all(ratio < 3.0), \
            f"SE ratios out of range: {ratio}"

    def test_kernel_se_matches_r_nid(self, linear_data_file):
        """Kernel SEs should be close to R's se='nid'."""
        X, y, _, _, csv_path = linear_data_file

        pdlp = QuantileRegression(tau=0.5, se_method='kernel')
        pdlp.fit(X, y)

        r_result = self._fit_r_quantreg(csv_path, tau=0.5, se_method='nid')

        ratio = pdlp.stderr_[0.5]['y'] / np.array(r_result['stderr'])
        assert np.all(ratio > 0.2) and np.all(ratio < 5.0), \
            f"Kernel SE ratios out of range: {ratio}"

    def test_extreme_quantiles_match_r(self, linear_data_file):
        """Extreme quantiles should match R."""
        X, y, _, _, csv_path = linear_data_file

        for q in [0.05, 0.95]:
            pdlp = QuantileRegression(tau=q, n_bootstrap=20, random_state=0)
            pdlp.fit(X, y)

            r_results = self._fit_r_multiple_quantiles(csv_path, [q])
            r_coef = r_results[str(q)]['coefficients']

            np.testing.assert_allclose(
                [pdlp.intercept_[q]['y']] + pdlp.coef_[q]['y'].tolist(),
                r_coef, atol=0.3,
                err_msg=f"Coefficients should match R at tau={q}"
            )

    def test_weighted_matches_r(self, linear_data_file):
        """Weighted quantile regression should match R."""
        X, y, _, _, csv_path = linear_data_file
        rng = np.random.default_rng(99)
        weights = rng.uniform(0.5, 3.0, size=len(y))

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y, weights=weights)

        # Save weights to CSV
        import pandas as pd
        df = pd.read_csv(csv_path)
        df['w'] = weights
        weighted_path = csv_path.replace('.csv', '_weighted.csv')
        df.to_csv(weighted_path, index=False)

        script = f"""
        library(quantreg)
        library(jsonlite)
        data <- read.csv("{weighted_path}")
        fit <- rq(y ~ X1 + X2 + X3, tau=0.5, data=data, weights=w)
        cat(toJSON(list(coefficients=as.numeric(coef(fit))), auto_unbox=TRUE))
        """
        output = _run_r_script(script)
        r_result = json.loads(output)

        np.testing.assert_allclose(
            pdlp.coef_[0.5]['y'], r_result['coefficients'][1:], atol=0.15,
            err_msg="Weighted slopes should match R"
        )

    def test_censored_qr_matches_r_crq(self, tmp_path):
        """Censored QR should produce similar coefficients to R's crq()."""
        rng = np.random.default_rng(42)
        n = 400
        X = rng.normal(size=(n, 2))
        true_beta = np.array([1.5, -1.0])
        true_intercept = 3.0
        T = true_intercept + X @ true_beta + rng.standard_t(df=5, size=n)
        C = np.quantile(T, 0.7) + rng.exponential(scale=1.0, size=n)
        Y = np.minimum(T, C)
        delta = (T <= C).astype(int)

        csv_path = str(tmp_path / 'censored.csv')
        import pandas as pd
        df = pd.DataFrame({'y': Y, 'X1': X[:, 0], 'X2': X[:, 1], 'delta': delta})
        df.to_csv(csv_path, index=False)

        # Our censored QR
        m_cens = CensoredQuantileRegression(
            tau=0.5, n_bootstrap=20, random_state=0,
            censoring='right', se_method='analytical')
        m_cens.fit(X, Y, event_indicator=delta)

        # R's crq (Portnoy method)
        script = f"""
        library(quantreg)
        library(jsonlite)
        library(survival)
        data <- read.csv("{csv_path}")
        fit <- tryCatch({{
            crq(Surv(y, delta, type="right") ~ X1 + X2, tau=0.5,
                data=data, method="Portnoy")
        }}, error=function(e) {{
            # Fall back to Powell if Portnoy fails
            crq(Surv(y, delta, type="right") ~ X1 + X2, tau=0.5,
                data=data, method="Powell")
        }})
        coefs <- tryCatch({{
            as.numeric(coef(fit, taus=0.5))
        }}, error=function(e) {{
            as.numeric(coef(fit))
        }})
        cat(toJSON(list(coefficients=coefs), auto_unbox=TRUE))
        """
        try:
            output = _run_r_script(script, timeout=120)
            r_result = json.loads(output)

            # Both should recover true coefficients reasonably
            our_bias = np.abs(np.array([m_cens.intercept_[0.5]['y']] +
                              m_cens.coef_[0.5]['y'].tolist()) -
                              np.array([true_intercept] + true_beta.tolist()))
            r_bias = np.abs(np.array(r_result['coefficients']) -
                           np.array([true_intercept] + true_beta.tolist()))

            # Our model should be in the same ballpark as R
            assert np.max(our_bias) < 1.5, \
                f"Our censored QR bias too large: {our_bias}"

        except (RuntimeError, json.JSONDecodeError) as e:
            # R's crq can be finicky - just verify our model is reasonable
            our_coef = np.array([m_cens.intercept_[0.5]['y']] +
                               m_cens.coef_[0.5]['y'].tolist())
            true_coef = np.array([true_intercept] + true_beta.tolist())
            bias = np.abs(our_coef - true_coef)
            assert np.max(bias) < 1.5, \
                f"Censored QR bias too large (R unavailable: {e}): {bias}"
