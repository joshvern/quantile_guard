# quantile_guard/quantile_regression.py

from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import linprog
from scipy import sparse
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed
import os
import time
import warnings


class QuantileRegression(BaseEstimator, RegressorMixin):
    """
    Quantile Regression using LP solvers from Google's OR-Tools, with statistical
    summaries, multi-output support, and advanced regularization.

    Parameters
    ----------
    tau : float or list of floats, default=0.5
        The quantile(s) to estimate, each must be between 0 and 1.

    n_bootstrap : int, default=1000
        Number of bootstrap samples for standard error estimation.
        Only used when ``se_method='bootstrap'``.

    random_state : int or None, default=None
        Seed for the random number generator.

    regularization : str, default='none'
        Regularization type: 'none', 'l1', 'elasticnet', 'scad', or 'mcp'.

    alpha : float, default=0.0
        Regularization strength. The objective is normalized as
        ``(1/n) * sum(pinball_loss) + alpha * penalty``, making alpha
        dataset-size-invariant and consistent with scikit-learn.

    l1_ratio : float, default=1.0
        Mixing parameter for elastic net (0 = pure L2, 1 = pure L1).
        Only used when ``regularization='elasticnet'``.

    n_jobs : int, default=1
        Number of parallel jobs for bootstrapping. ``-1`` uses all cores.

    solver_backend : str, default='PDLP'
        LP solver: 'PDLP' (first-order, large-scale) or 'GLOP' (simplex).

    solver_tol : float or None, default=None
        Solver optimality tolerance. If None, uses solver default.

    solver_time_limit : float or None, default=None
        Maximum solver time in seconds.

    enforce_non_crossing_predict : bool, default=True
        Enforce monotonic quantile predictions via row-wise rearrangement
        when prediction-time crossings are detected.

    se_method : str, default='bootstrap'
        Standard error method: 'bootstrap', 'analytical' (IID), or 'kernel'
        (heteroscedasticity-robust sandwich).

    use_sparse : bool, default=False
        Use scipy sparse LP solver instead of OR-Tools. More memory-efficient
        for large datasets.

    Attributes
    ----------
    coef_ : dict
        Coefficients ``{tau: {output: array}}``.
    intercept_ : dict
        Intercepts ``{tau: {output: float}}``.
    stderr_ : dict
        Standard errors ``{tau: {output: array}}``.
    tvalues_ : dict
        T-statistics ``{tau: {output: array}}``.
    pvalues_ : dict
        P-values ``{tau: {output: array}}``.
    confidence_intervals_ : dict
        95% bootstrap CIs ``{tau: {output: DataFrame}}``.
    solver_info_ : dict
        Solver diagnostics from the most recent fit.
    feature_names_ : list
        Feature names.
    output_names_ : list
        Output names.
    """

    def __init__(self, tau=0.5, n_bootstrap=1000, random_state=None,
                 regularization='none', alpha=0.0, l1_ratio=1.0,
                 n_jobs=1, solver_backend='PDLP', solver_tol=None,
                 solver_time_limit=None, enforce_non_crossing_predict=True,
                 se_method='bootstrap', use_sparse=False):
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        self.solver_backend = solver_backend
        self.solver_tol = solver_tol
        self.solver_time_limit = solver_time_limit
        self.enforce_non_crossing_predict = enforce_non_crossing_predict
        self.se_method = se_method
        self.use_sparse = use_sparse

        self.coef_ = None
        self.intercept_ = None
        self.stderr_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.confidence_intervals_ = None
        self.solver_info_ = None
        self.feature_names_ = None
        self.output_names_ = None
        self._is_fitted = None
        self._pinball_loss_ = None
        self._null_pinball_loss_ = None
        self._X_aug = None
        self._y = None
        self._weights = None

    # ================================================================
    # Core fitting
    # ================================================================

    def fit(self, X, y, weights=None, clusters=None):
        """
        Fit the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        weights : array-like of shape (n_samples,), optional
            Observation weights.
        clusters : array-like of shape (n_samples,), optional
            Cluster labels for cluster-robust standard errors.

        Returns
        -------
        self
        """
        X, y, weights = self._validate_inputs(X, y, weights)
        self._validate_tau()
        if clusters is not None and len(np.asarray(clusters)) != X.shape[0]:
            raise ValueError(
                "clusters must have the same length as the number of observations."
            )

        n_samples, n_features = X.shape
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        self._init_storage(n_features)
        self._X_aug = X_augmented
        self._y = y
        self._weights = weights

        # Estimate coefficients
        self._fit_coefficients(X_augmented, y, weights)

        # Compute pinball losses for pseudo R²
        self._compute_pinball_losses(X_augmented, y, weights)

        # Compute standard errors
        if clusters is not None:
            self._compute_cluster_se(X_augmented, y, weights, np.asarray(clusters))
        elif self.se_method == 'bootstrap':
            self._compute_bootstrap_se(X_augmented, y, weights)
        elif self.se_method in ('analytical', 'kernel'):
            self._compute_analytical_se(X_augmented, y, method=self.se_method)

        for q in self.tau:
            for output in self.output_names_:
                self._is_fitted[q][output] = True

        return self

    def fit_formula(self, formula, data, weights=None, clusters=None):
        """
        Fit using an R-style formula (requires ``patsy``).

        Parameters
        ----------
        formula : str
            R-style formula, e.g. ``"y ~ x1 + x2 + C(group)"``.
        data : DataFrame
            Data containing all variables referenced in the formula.
        weights : str or array-like, optional
            Column name in data or array of weights.
        clusters : str or array-like, optional
            Column name in data or array of cluster labels.

        Returns
        -------
        self
        """
        try:
            import patsy
        except ImportError:
            raise ImportError(
                "Formula interface requires 'patsy'. "
                "Install with: pip install patsy"
            )

        y_df, X_df = patsy.dmatrices(formula, data, return_type='dataframe')

        # Drop patsy's intercept (we add our own)
        if 'Intercept' in X_df.columns:
            X_df = X_df.drop('Intercept', axis=1)

        w = data[weights].values if isinstance(weights, str) else weights
        c = data[clusters].values if isinstance(clusters, str) else clusters

        self._formula = formula
        return self.fit(X_df, y_df, weights=w, clusters=c)

    def _validate_inputs(self, X, y, weights):
        """Validate and convert inputs to numpy arrays."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            X = np.asarray(X)
            self.feature_names_ = [f'X{i}' for i in range(1, X.shape[1] + 1)]

        if isinstance(y, pd.DataFrame):
            self.output_names_ = y.columns.tolist()
            y = y.values
        elif isinstance(y, pd.Series):
            self.output_names_ = [y.name if y.name is not None else 'y']
            y = y.values.reshape(-1, 1)
        else:
            y = np.asarray(y)
            if y.ndim == 1:
                self.output_names_ = ['y']
                y = y.reshape(-1, 1)
            elif y.ndim == 2:
                self.output_names_ = [f'y{i}' for i in range(1, y.shape[1] + 1)]
            else:
                raise ValueError("y must be a 1D or 2D array-like structure.")

        n_samples = X.shape[0]
        if weights is None:
            weights = np.ones(n_samples)
        else:
            weights = np.asarray(weights)
            if weights.shape[0] != n_samples:
                raise ValueError(
                    "Weights array must have the same length as the number of observations."
                )

        return X, y, weights

    def _init_storage(self, n_features):
        """Initialize coefficient and inference storage."""
        self.coef_ = {q: {o: np.zeros(n_features) for o in self.output_names_} for q in self.tau}
        self.intercept_ = {q: {o: 0.0 for o in self.output_names_} for q in self.tau}
        self.stderr_ = {q: {o: np.zeros(n_features + 1) for o in self.output_names_} for q in self.tau}
        self.tvalues_ = {q: {o: np.zeros(n_features + 1) for o in self.output_names_} for q in self.tau}
        self.pvalues_ = {q: {o: np.zeros(n_features + 1) for o in self.output_names_} for q in self.tau}
        self.confidence_intervals_ = {q: {o: None for o in self.output_names_} for q in self.tau}
        self._is_fitted = {q: {o: False for o in self.output_names_} for q in self.tau}
        self._pinball_loss_ = {q: {} for q in self.tau}
        self._null_pinball_loss_ = {q: {} for q in self.tau}

    def _validate_tau(self):
        """Validate and sort tau."""
        if isinstance(self.tau, (int, float)):
            if not 0 < self.tau < 1:
                raise ValueError("Each quantile tau must be between 0 and 1.")
            self.tau = [float(self.tau)]
        elif isinstance(self.tau, list):
            if not all(isinstance(q, (int, float)) and 0 < q < 1 for q in self.tau):
                raise ValueError("All quantiles tau must be floats between 0 and 1.")
            if len(set(self.tau)) != len(self.tau):
                raise ValueError("Quantile tau values must be unique.")
            self.tau = sorted([float(q) for q in self.tau])
        else:
            raise TypeError("tau must be a float or a list of floats.")

    def _fit_coefficients(self, X, y, weights):
        """Estimate regression coefficients (without standard errors)."""
        if self.regularization in ('scad', 'mcp', 'elasticnet'):
            coefficients, solver_info = self._solve_with_lla(X, y, weights)
        else:
            coefficients, solver_info = self._solve_lp(X, y, weights)

        self.solver_info_ = solver_info

        for q in self.tau:
            for idx, output in enumerate(self.output_names_):
                self.intercept_[q][output] = coefficients[q][idx][0]
                self.coef_[q][output] = coefficients[q][idx][1:]

    def _compute_pinball_losses(self, X, y, weights):
        """Compute fitted and null pinball losses for pseudo R²."""
        weight_sum = np.sum(weights)
        for q in self.tau:
            for ki, output in enumerate(self.output_names_):
                coef_full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                fitted = X @ coef_full
                resid = y[:, ki] - fitted
                loss = np.sum(weights * np.where(resid >= 0, q * resid, (q - 1) * resid)) / weight_sum
                self._pinball_loss_[q][output] = loss

                null_val = np.quantile(y[:, ki], q)
                null_resid = y[:, ki] - null_val
                null_loss = np.sum(weights * np.where(null_resid >= 0, q * null_resid, (q - 1) * null_resid)) / weight_sum
                self._null_pinball_loss_[q][output] = null_loss

    # ================================================================
    # LP Solvers
    # ================================================================

    def _solve_lp(self, X, y, weights, penalty_weights=None, return_coefficients=True):
        """Dispatch to OR-Tools or scipy solver."""
        if self.use_sparse:
            return self._solve_scipy_lp(X, y, weights, penalty_weights, return_coefficients)
        return self._solve_ortools_lp(X, y, weights, penalty_weights, return_coefficients)

    def _create_solver(self):
        """Create and configure an OR-Tools LP solver."""
        solver = pywraplp.Solver.CreateSolver(self.solver_backend)
        if not solver:
            raise ValueError(
                f"Solver '{self.solver_backend}' is not available. "
                f"Try 'PDLP' or 'GLOP'."
            )
        if self.solver_time_limit is not None:
            solver.SetTimeLimit(int(self.solver_time_limit * 1000))
        if self.solver_tol is not None and self.solver_backend == 'PDLP':
            try:
                solver.SetSolverSpecificParametersAsString(
                    f'termination_criteria {{ eps_optimal_absolute: {self.solver_tol} '
                    f'eps_optimal_relative: {self.solver_tol} }}'
                )
            except Exception:
                pass
        return solver

    def _solve_ortools_lp(self, X, y, weights, penalty_weights=None, return_coefficients=True):
        """Solve quantile regression LP using OR-Tools."""
        n_samples, n_features_plus_1 = X.shape
        n_outputs = y.shape[1]
        weight_sum = np.sum(weights)

        solver = self._create_solver()
        infinity = solver.infinity()

        has_penalty = (penalty_weights is not None) or (self.regularization == 'l1' and self.alpha > 0)

        beta = {q: {k: [solver.NumVar(-infinity, infinity, f'b_{j}_q{q}_k{k}')
                       for j in range(n_features_plus_1)] for k in range(n_outputs)}
                for q in self.tau}
        r_pos = {q: {k: [solver.NumVar(0, infinity, f'rp_{i}_q{q}_k{k}')
                         for i in range(n_samples)] for k in range(n_outputs)}
                  for q in self.tau}
        r_neg = {q: {k: [solver.NumVar(0, infinity, f'rn_{i}_q{q}_k{k}')
                         for i in range(n_samples)] for k in range(n_outputs)}
                  for q in self.tau}

        if has_penalty:
            z = {q: {k: [solver.NumVar(0, infinity, f'z_{j}_q{q}_k{k}')
                        for j in range(n_features_plus_1 - 1)] for k in range(n_outputs)}
                 for q in self.tau}
            for q in self.tau:
                for k in range(n_outputs):
                    for j in range(1, n_features_plus_1):
                        solver.Add(beta[q][k][j] <= z[q][k][j - 1])
                        solver.Add(-beta[q][k][j] <= z[q][k][j - 1])

        objective = solver.Objective()
        for q in self.tau:
            for k in range(n_outputs):
                for i in range(n_samples):
                    constraint_expr = (
                        sum(X[i, j] * beta[q][k][j] for j in range(n_features_plus_1)) +
                        r_pos[q][k][i] - r_neg[q][k][i]
                    )
                    solver.Add(constraint_expr == y[i, k])
                    objective.SetCoefficient(r_pos[q][k][i], q * weights[i] / weight_sum)
                    objective.SetCoefficient(r_neg[q][k][i], (1 - q) * weights[i] / weight_sum)

        if has_penalty:
            for q in self.tau:
                for k in range(n_outputs):
                    for j in range(n_features_plus_1 - 1):
                        if penalty_weights is not None:
                            w = penalty_weights[q][k][j]
                        else:
                            w = self.alpha
                        objective.SetCoefficient(z[q][k][j], w)

        objective.SetMinimization()

        # Non-crossing constraints
        for k in range(n_outputs):
            for i in range(n_samples):
                for q_idx in range(len(self.tau) - 1):
                    q_lo = self.tau[q_idx]
                    q_hi = self.tau[q_idx + 1]
                    pred_lo = sum(X[i, j] * beta[q_lo][k][j] for j in range(n_features_plus_1))
                    pred_hi = sum(X[i, j] * beta[q_hi][k][j] for j in range(n_features_plus_1))
                    solver.Add(pred_lo <= pred_hi)

        wall_start = time.perf_counter()
        status = solver.Solve()
        wall_time = time.perf_counter() - wall_start

        status_map = {
            pywraplp.Solver.OPTIMAL: 'OPTIMAL',
            pywraplp.Solver.FEASIBLE: 'FEASIBLE',
            pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
            pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
            pywraplp.Solver.ABNORMAL: 'ABNORMAL',
            pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
        }
        solver_info = {
            'status': status_map.get(status, f'UNKNOWN({status})'),
            'wall_time_seconds': round(wall_time, 4),
            'num_variables': solver.NumVariables(),
            'num_constraints': solver.NumConstraints(),
            'objective_value': solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else None,
        }

        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(
                f"Solver did not find an optimal solution. "
                f"Status: {solver_info['status']}. "
                f"Try adjusting solver_tol, solver_time_limit, or solver_backend."
            )

        if return_coefficients:
            coefficients = {}
            for q in self.tau:
                coefficients[q] = {}
                for k in range(n_outputs):
                    vals = np.array([beta[q][k][j].solution_value() for j in range(n_features_plus_1)])
                    coefficients[q][k] = vals
            return coefficients, solver_info
        return None, solver_info

    def _solve_scipy_lp(self, X, y, weights, penalty_weights=None, return_coefficients=True):
        """Solve quantile regression LP using scipy sparse matrices + HiGHS."""
        n, p = X.shape
        K = y.shape[1]
        Q = len(self.tau)
        weight_sum = np.sum(weights)

        has_penalty = (penalty_weights is not None) or (self.regularization == 'l1' and self.alpha > 0)

        n_beta = Q * K * p
        n_rpos = Q * K * n
        n_rneg = Q * K * n
        n_z = Q * K * (p - 1) if has_penalty else 0
        n_vars = n_beta + n_rpos + n_rneg + n_z

        # ---- Objective ----
        c = np.zeros(n_vars)
        for qi, q in enumerate(self.tau):
            for ki in range(K):
                base_rp = n_beta + (qi * K + ki) * n
                base_rn = n_beta + n_rpos + (qi * K + ki) * n
                c[base_rp:base_rp + n] = q * weights / weight_sum
                c[base_rn:base_rn + n] = (1 - q) * weights / weight_sum

        if has_penalty:
            for qi, q in enumerate(self.tau):
                for ki in range(K):
                    base_z = n_beta + n_rpos + n_rneg + (qi * K + ki) * (p - 1)
                    for j in range(p - 1):
                        if penalty_weights is not None:
                            c[base_z + j] = penalty_weights[q][ki][j]
                        else:
                            c[base_z + j] = self.alpha

        # ---- Equality constraints: X @ beta + r_pos - r_neg = y ----
        n_eq = Q * K * n
        qk = Q * K
        identity_qk = sparse.eye(qk, format='csc')
        identity_n = sparse.eye(n, format='csc')

        beta_block = sparse.kron(identity_qk, sparse.csc_matrix(X), format='csc')
        rpos_block = sparse.kron(identity_qk, identity_n, format='csc')
        rneg_block = sparse.kron(identity_qk, -identity_n, format='csc')

        eq_blocks = [beta_block, rpos_block, rneg_block]
        if has_penalty:
            eq_blocks.append(sparse.csc_matrix((n_eq, n_z)))
        A_eq = sparse.hstack(eq_blocks, format='csc')
        b_eq = np.tile(y.T.reshape(-1), Q)

        # ---- Inequality constraints ----
        ub_rows, ub_cols, ub_data, b_ub_parts = [], [], [], []
        ub_row = 0

        # L1: beta_j <= z_j and -beta_j <= z_j
        if has_penalty:
            for qi, q in enumerate(self.tau):
                for ki in range(K):
                    beta_off = (qi * K + ki) * p
                    z_off = n_beta + n_rpos + n_rneg + (qi * K + ki) * (p - 1)
                    for j in range(p - 1):
                        # beta_{j+1} - z_j <= 0
                        ub_rows.append(np.array([ub_row, ub_row], dtype=int))
                        ub_cols.append(np.array([beta_off + j + 1, z_off + j], dtype=int))
                        ub_data.append(np.array([1.0, -1.0]))
                        b_ub_parts.append(0.0)
                        ub_row += 1
                        # -beta_{j+1} - z_j <= 0
                        ub_rows.append(np.array([ub_row, ub_row], dtype=int))
                        ub_cols.append(np.array([beta_off + j + 1, z_off + j], dtype=int))
                        ub_data.append(np.array([-1.0, -1.0]))
                        b_ub_parts.append(0.0)
                        ub_row += 1

        # Non-crossing
        x_flat = X.reshape(-1)
        nonzero_mask = x_flat != 0
        row_pattern = np.repeat(np.arange(n), p)[nonzero_mask]
        col_pattern = np.tile(np.arange(p), n)[nonzero_mask]
        data_pattern = x_flat[nonzero_mask]

        for ki in range(K):
            for qi in range(Q - 1):
                beta_lo_off = (qi * K + ki) * p
                beta_hi_off = ((qi + 1) * K + ki) * p
                if data_pattern.size:
                    ub_rows.append(ub_row + row_pattern)
                    ub_cols.append(beta_lo_off + col_pattern)
                    ub_data.append(data_pattern)

                    ub_rows.append(ub_row + row_pattern)
                    ub_cols.append(beta_hi_off + col_pattern)
                    ub_data.append(-data_pattern)
                b_ub_parts.extend(np.zeros(n))
                ub_row += n

        if ub_row > 0:
            ub_row_idx = np.concatenate(ub_rows) if ub_rows else np.array([], dtype=int)
            ub_col_idx = np.concatenate(ub_cols) if ub_cols else np.array([], dtype=int)
            ub_values = np.concatenate(ub_data) if ub_data else np.array([], dtype=float)
            A_ub = sparse.csc_matrix(
                (ub_values, (ub_row_idx, ub_col_idx)),
                shape=(ub_row, n_vars)
            )
            b_ub = np.array(b_ub_parts)
        else:
            A_ub, b_ub = None, None

        # Bounds
        bounds = [(None, None)] * n_beta + [(0, None)] * (n_rpos + n_rneg + n_z)

        wall_start = time.perf_counter()
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
        except Exception:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='interior-point')
        wall_time = time.perf_counter() - wall_start

        solver_info = {
            'status': 'OPTIMAL' if result.success else result.message,
            'wall_time_seconds': round(wall_time, 4),
            'num_variables': n_vars,
            'num_constraints': n_eq + (ub_row if ub_row else 0),
            'objective_value': result.fun if result.success else None,
        }

        if not result.success:
            raise RuntimeError(f"Scipy LP solver failed: {result.message}")

        if return_coefficients:
            coefficients = {}
            for qi, q in enumerate(self.tau):
                coefficients[q] = {}
                for ki in range(K):
                    start = (qi * K + ki) * p
                    coefficients[q][ki] = result.x[start:start + p]
            return coefficients, solver_info
        return None, solver_info

    # ================================================================
    # Regularization: LLA for SCAD / MCP / Elastic Net
    # ================================================================

    def _solve_with_lla(self, X, y, weights, max_iter=20, tol=1e-4):
        """Solve via Local Linear Approximation for non-convex / elastic net penalties."""
        n_outputs = y.shape[1]
        p = X.shape[1]

        # Initial solution from L1
        init_pw = {q: {k: np.full(p - 1, self.alpha * self.l1_ratio)
                       for k in range(n_outputs)} for q in self.tau}
        coefficients, solver_info = self._solve_lp(X, y, weights, penalty_weights=init_pw)

        for iteration in range(max_iter):
            prev = {q: {k: coefficients[q][k].copy() for k in coefficients[q]}
                    for q in coefficients}

            pw = self._compute_penalty_weights(coefficients, p)
            coefficients, solver_info = self._solve_lp(X, y, weights, penalty_weights=pw)

            max_change = max(
                np.max(np.abs(coefficients[q][k] - prev[q][k]))
                for q in coefficients for k in coefficients[q]
            )
            if max_change < tol:
                break

        return coefficients, solver_info

    def _compute_penalty_weights(self, coefficients, p):
        """Compute adaptive penalty weights for LLA."""
        pw = {}
        eps = 1e-6
        for q in self.tau:
            pw[q] = {}
            for k in coefficients[q]:
                beta_abs = np.abs(coefficients[q][k][1:])  # exclude intercept

                if self.regularization == 'elasticnet':
                    w = self.alpha * self.l1_ratio + self.alpha * (1 - self.l1_ratio) * beta_abs
                elif self.regularization == 'scad':
                    a = 3.7
                    w = np.where(
                        beta_abs <= self.alpha,
                        self.alpha,
                        np.where(
                            beta_abs <= a * self.alpha,
                            np.maximum(a * self.alpha - beta_abs, 0) / (a - 1),
                            0.0
                        )
                    )
                elif self.regularization == 'mcp':
                    gamma = 3.0
                    w = np.maximum(self.alpha - beta_abs / gamma, 0.0)
                else:
                    w = np.full(len(beta_abs), self.alpha)

                pw[q][k] = w
        return pw

    # ================================================================
    # Inference: Bootstrap / Analytical / Cluster-Robust
    # ================================================================

    def _compute_bootstrap_se(self, X, y, weights):
        """Bootstrap standard errors with empirical p-values and CIs."""
        n_samples, n_features_plus_1 = X.shape
        n_outputs = y.shape[1]
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(
            np.iinfo(np.uint32).max,
            size=self.n_bootstrap,
            dtype=np.uint32,
        )
        n_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
        n_jobs = max(int(n_jobs or 1), 1)

        if n_jobs == 1:
            results = [
                self._fit_bootstrap_sample(
                    X, y, weights, int(seed), n_outputs, n_features_plus_1
                )
                for seed in tqdm(seeds, total=self.n_bootstrap, desc='Bootstrapping')
            ]
        else:
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(self._fit_bootstrap_sample)(
                    X, y, weights, int(seed), n_outputs, n_features_plus_1
                )
                for seed in seeds
            )

        beta_bootstrap = np.stack(results, axis=0)

        for qi, q in enumerate(self.tau):
            for ki, output in enumerate(self.output_names_):
                samples = beta_bootstrap[:, qi, ki, :]
                valid = samples[~np.isnan(samples).any(axis=1)]

                if valid.shape[0] == 0:
                    raise RuntimeError(
                        f"All bootstrap iterations failed for quantile {q}, output {output}.")

                stderr = np.std(valid, axis=0, ddof=1)
                self.stderr_[q][output] = stderr

                # Empirical p-values
                pvalues = np.zeros(n_features_plus_1)
                for j in range(n_features_plus_1):
                    p_pos = np.mean(valid[:, j] > 0)
                    p_neg = np.mean(valid[:, j] < 0)
                    pvalues[j] = np.clip(2 * min(p_pos, p_neg), 0, 1)
                self.pvalues_[q][output] = pvalues

                coef_full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.tvalues_[q][output] = np.where(stderr > 0, coef_full / stderr, 0.0)

                ci_lo = np.percentile(valid, 2.5, axis=0)
                ci_hi = np.percentile(valid, 97.5, axis=0)
                index = ['Intercept'] + self.feature_names_
                self.confidence_intervals_[q][output] = pd.DataFrame({
                    'lower_2.5%': ci_lo, 'upper_97.5%': ci_hi
                }, index=index)

    def _fit_bootstrap_sample(self, X, y, weights, seed, n_outputs, n_features_plus_1):
        """Fit a single bootstrap resample and return coefficients as an array."""
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, X.shape[0], size=X.shape[0])
        try:
            beta_sample, _ = self._solve_lp(
                X[idx], y[idx], weights[idx], return_coefficients=True
            )
        except Exception:
            return np.full((len(self.tau), n_outputs, n_features_plus_1), np.nan)

        coef_array = np.empty((len(self.tau), n_outputs, n_features_plus_1))
        for qi, q in enumerate(self.tau):
            for ki in range(n_outputs):
                coef_array[qi, ki, :] = beta_sample[q][ki]
        return coef_array

    def _compute_analytical_se(self, X, y, method='analytical'):
        """
        Analytical standard errors using asymptotic sandwich estimator.

        Parameters
        ----------
        method : str
            'analytical' for IID (Koenker-Bassett 1978) or
            'kernel' for heteroscedasticity-robust (Powell 1991).
        """
        n, p = X.shape
        XtX = X.T @ X
        XtX_inv = self._safe_inverse(XtX)

        for q in self.tau:
            for ki, output in enumerate(self.output_names_):
                coef_full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                residuals = y[:, ki] - X @ coef_full

                if method == 'analytical':
                    # IID: V = tau*(1-tau) * s^2 * (X'X)^{-1}
                    sparsity = self._estimate_sparsity(residuals, q, n)
                    cov = q * (1 - q) * sparsity ** 2 * XtX_inv
                else:
                    # Kernel sandwich: V = H^{-1} J H^{-1} / n
                    h = self._hall_sheather_bandwidth(q, n)
                    fhat = norm.pdf(residuals / h) / h
                    H = X.T @ (X * fhat[:, None]) / n
                    J = q * (1 - q) * XtX / n
                    H_inv = self._safe_inverse(H)
                    cov = H_inv @ J @ H_inv / n

                stderr = np.sqrt(np.maximum(np.diag(cov), 0))
                self.stderr_[q][output] = stderr

                with np.errstate(divide='ignore', invalid='ignore'):
                    tvals = np.where(stderr > 0, coef_full / stderr, 0.0)
                self.tvalues_[q][output] = tvals

                df = n - p
                pvals = 2 * (1 - t.cdf(np.abs(tvals), df=df))
                self.pvalues_[q][output] = pvals

                index = ['Intercept'] + self.feature_names_
                # Approximate CIs from normal distribution
                z_crit = norm.ppf(0.975)
                self.confidence_intervals_[q][output] = pd.DataFrame({
                    'lower_2.5%': coef_full - z_crit * stderr,
                    'upper_97.5%': coef_full + z_crit * stderr,
                }, index=index)

    def _compute_cluster_se(self, X, y, weights, clusters):
        """
        Cluster-robust standard errors via clustered sandwich estimator.
        """
        n, p = X.shape
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        if G < 2:
            raise ValueError("Cluster-robust standard errors require at least 2 clusters.")

        cluster_index = {g: idx for idx, g in enumerate(unique_clusters)}
        cluster_codes = np.array([cluster_index[g] for g in clusters], dtype=int)

        for q in self.tau:
            for ki, output in enumerate(self.output_names_):
                coef_full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                residuals = y[:, ki] - X @ coef_full

                # Bread: H = (1/n) X' D X where D = diag(f_i(0))
                h = self._hall_sheather_bandwidth(q, n)
                fhat = norm.pdf(residuals / h) / h
                H = X.T @ (X * fhat[:, None]) / n
                H_inv = self._safe_inverse(H)

                # Meat: J_cluster = (1/n) sum_g (sum_{i in g} psi_i x_i)(sum_{i in g} psi_i x_i)'
                psi = q - (residuals < 0).astype(float)  # subgradient
                score_rows = X * (psi * weights)[:, None]
                cluster_scores = np.zeros((G, p))
                np.add.at(cluster_scores, cluster_codes, score_rows)
                meat = cluster_scores.T @ cluster_scores
                meat /= n

                # Small-sample correction
                correction = G / (G - 1) * (n - 1) / (n - p)

                cov = correction * H_inv @ meat @ H_inv / n
                stderr = np.sqrt(np.maximum(np.diag(cov), 0))
                self.stderr_[q][output] = stderr

                with np.errstate(divide='ignore', invalid='ignore'):
                    tvals = np.where(stderr > 0, coef_full / stderr, 0.0)
                self.tvalues_[q][output] = tvals

                df = G - 1
                pvals = 2 * (1 - t.cdf(np.abs(tvals), df=df))
                self.pvalues_[q][output] = pvals

                index = ['Intercept'] + self.feature_names_
                t_crit = t.ppf(0.975, df=df)
                self.confidence_intervals_[q][output] = pd.DataFrame({
                    'lower_2.5%': coef_full - t_crit * stderr,
                    'upper_97.5%': coef_full + t_crit * stderr,
                }, index=index)

    @staticmethod
    def _safe_inverse(matrix):
        """Invert a matrix, falling back to the pseudo-inverse if needed."""
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    def _estimate_sparsity(self, residuals, tau, n):
        """Estimate sparsity function s(tau) = 1/f(F^{-1}(tau)) via order statistics."""
        h = self._hall_sheather_bandwidth(tau, n)
        tau_lo = max(tau - h, 0.5 / n)
        tau_hi = min(tau + h, 1 - 0.5 / n)

        sorted_res = np.sort(residuals)
        idx_lo = max(0, int(np.floor(n * tau_lo)))
        idx_hi = min(n - 1, int(np.ceil(n * tau_hi)))
        if idx_hi <= idx_lo:
            idx_hi = min(idx_lo + 1, n - 1)

        sparsity = (sorted_res[idx_hi] - sorted_res[idx_lo]) / (2 * h)
        return max(sparsity, 1e-10)

    @staticmethod
    def _hall_sheather_bandwidth(tau, n, alpha=0.05):
        """Hall-Sheather bandwidth for sparsity estimation."""
        z_tau = norm.ppf(tau)
        z_alpha = norm.ppf(1 - alpha / 2)
        phi_tau = norm.pdf(z_tau)
        return n ** (-1 / 3) * z_alpha ** (2 / 3) * (
            1.5 * phi_tau ** 2 / (2 * z_tau ** 2 + 1)
        ) ** (1 / 3)

    # ================================================================
    # Prediction
    # ================================================================

    def predict(self, X):
        """
        Predict using the fitted model.

        Returns
        -------
        y_pred : dict
            ``{tau: {output: array}}``.
        """
        if not all(all(self._is_fitted[q].values()) for q in self.tau):
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)

        X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

        y_pred = {q: {} for q in self.tau}
        for output in self.output_names_:
            coef_matrix = np.column_stack([
                np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                for q in self.tau
            ])
            pred_mat = X_aug @ coef_matrix
            if self.enforce_non_crossing_predict and len(self.tau) > 1:
                pred_mat = np.sort(pred_mat, axis=1)
            for idx, q in enumerate(self.tau):
                y_pred[q][output] = pred_mat[:, idx]

        return y_pred

    def predict_interval(self, X, coverage=0.90):
        """
        Predict intervals using fitted quantiles.

        Uses the fitted quantiles closest to ``(1-coverage)/2`` and
        ``1-(1-coverage)/2`` as lower/upper bounds.

        Parameters
        ----------
        X : array-like
            Samples.
        coverage : float, default=0.90
            Desired coverage probability.

        Returns
        -------
        result : dict
            ``{output: {'lower': array, 'median': array, 'upper': array}}``.
        """
        if len(self.tau) < 2:
            raise ValueError("predict_interval requires at least 2 fitted quantiles.")
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")

        target_lo = (1 - coverage) / 2
        target_hi = 1 - target_lo
        target_med = 0.5

        tau_lo = min(self.tau, key=lambda t: abs(t - target_lo))
        tau_hi = min(self.tau, key=lambda t: abs(t - target_hi))
        tau_med = min(self.tau, key=lambda t: abs(t - target_med))

        y_pred = self.predict(X)

        result = {}
        for output in self.output_names_:
            result[output] = {
                'lower': y_pred[tau_lo][output],
                'median': y_pred[tau_med][output],
                'upper': y_pred[tau_hi][output],
                'tau_lower': tau_lo,
                'tau_median': tau_med,
                'tau_upper': tau_hi,
            }
        return result

    # ================================================================
    # Model evaluation
    # ================================================================

    def score(self, X, y, sample_weight=None):
        """
        Return negative mean pinball loss (higher is better, sklearn convention).
        """
        y_pred = self.predict(X)

        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        else:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        total_loss = 0.0
        count = 0
        for q in self.tau:
            for ki, output in enumerate(self.output_names_):
                resid = y[:, ki] - y_pred[q][output]
                loss = np.where(resid >= 0, q * resid, (q - 1) * resid)
                if sample_weight is not None:
                    total_loss += np.average(loss, weights=sample_weight)
                else:
                    total_loss += np.mean(loss)
                count += 1

        return -total_loss / count

    @property
    def pseudo_r_squared_(self):
        """
        Koenker-Machado (1999) pseudo R-squared.

        ``R1(tau) = 1 - fitted_pinball_loss / null_pinball_loss``

        Returns
        -------
        dict : ``{tau: {output: float}}``
        """
        if self._pinball_loss_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            q: {
                o: 1 - self._pinball_loss_[q][o] / self._null_pinball_loss_[q][o]
                if self._null_pinball_loss_[q][o] > 0 else 0.0
                for o in self.output_names_
            }
            for q in self.tau
        }

    def summary(self):
        """
        Summary tables with coefficients, SEs, t-values, p-values, and CIs.

        Returns
        -------
        dict : ``{tau: {output: DataFrame}}``
        """
        if not all(all(self._is_fitted[q].values()) for q in self.tau):
            raise RuntimeError("Model is not fitted yet.")

        summary_dict = {}
        for q in self.tau:
            summary_dict[q] = {}
            for output in self.output_names_:
                coef = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                index = ['Intercept'] + self.feature_names_
                ci = self.confidence_intervals_[q][output]
                data = {
                    'Coefficient': coef,
                    'Std. Error': self.stderr_[q][output],
                    't-value': self.tvalues_[q][output],
                    'P>|t|': self.pvalues_[q][output],
                }
                if ci is not None:
                    data['CI 2.5%'] = ci['lower_2.5%'].values
                    data['CI 97.5%'] = ci['upper_97.5%'].values
                summary_dict[q][output] = pd.DataFrame(data, index=index)
        return summary_dict

    # ================================================================
    # Visualization
    # ================================================================

    def plot_quantile_process(self, feature=None, figsize=None, ax=None):
        """
        Plot coefficient estimates across quantiles with confidence intervals.

        Parameters
        ----------
        feature : str, list of str, or None
            Feature(s) to plot. None plots all features.
        figsize : tuple, optional
            Figure size.
        ax : matplotlib Axes, optional
            Axes to plot on (single feature only).

        Returns
        -------
        fig : matplotlib Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install: pip install matplotlib")

        if len(self.tau) < 2:
            raise ValueError("Quantile process plot requires at least 2 fitted quantiles.")

        if feature is None:
            features = ['Intercept'] + self.feature_names_
        elif isinstance(feature, str):
            features = [feature]
        else:
            features = list(feature)

        all_names = ['Intercept'] + self.feature_names_
        n_features = len(features)
        n_outputs = len(self.output_names_)
        n_plots = n_features * n_outputs

        if ax is not None:
            axes = [ax] * n_plots
            fig = ax.figure
        else:
            ncols = min(n_features, 3)
            nrows = int(np.ceil(n_plots / ncols))
            fig, axes_arr = plt.subplots(nrows, ncols,
                                         figsize=figsize or (5 * ncols, 4 * nrows),
                                         squeeze=False)
            axes = axes_arr.flatten()

        plot_idx = 0
        for output in self.output_names_:
            for feat in features:
                cur_ax = axes[plot_idx]
                feat_idx = all_names.index(feat)

                coefs, ci_lo, ci_hi = [], [], []
                for q in self.tau:
                    full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                    coefs.append(full[feat_idx])
                    ci = self.confidence_intervals_[q][output]
                    if ci is not None:
                        ci_lo.append(ci.iloc[feat_idx, 0])
                        ci_hi.append(ci.iloc[feat_idx, 1])

                cur_ax.plot(self.tau, coefs, 'b-o', markersize=4, label='QR estimate')
                if ci_lo:
                    cur_ax.fill_between(self.tau, ci_lo, ci_hi, alpha=0.2, color='blue',
                                        label='95% CI')
                cur_ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                cur_ax.set_xlabel('\u03c4')
                cur_ax.set_ylabel('Coefficient')
                title = feat if n_outputs == 1 else f'{feat} ({output})'
                cur_ax.set_title(title)
                cur_ax.legend(fontsize=8)
                plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        return fig

    # ================================================================
    # Scikit-learn API
    # ================================================================

    def get_params(self, deep=True):
        return {
            'tau': self.tau,
            'n_bootstrap': self.n_bootstrap,
            'random_state': self.random_state,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'n_jobs': self.n_jobs,
            'solver_backend': self.solver_backend,
            'solver_tol': self.solver_tol,
            'solver_time_limit': self.solver_time_limit,
            'enforce_non_crossing_predict': self.enforce_non_crossing_predict,
            'se_method': self.se_method,
            'use_sparse': self.use_sparse,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ====================================================================
# Censored Quantile Regression
# ====================================================================

class CensoredQuantileRegression(QuantileRegression):
    """
    Censored (survival) quantile regression via iterative reweighting.

    Implements Powell (1986) iterative algorithm for right- or left-censored data.
    At each iteration, censored observations are included or excluded based on
    whether the estimated conditional quantile exceeds the censoring point.

    Parameters
    ----------
    censoring : str, default='right'
        Type of censoring: 'right' or 'left'.
    max_censor_iter : int, default=50
        Maximum number of censored QR iterations.
    censor_tol : float, default=1e-4
        Convergence tolerance for coefficient changes between iterations.
    **kwargs
        Additional parameters passed to QuantileRegression.

    Examples
    --------
    >>> model = CensoredQuantileRegression(tau=0.5, censoring='right',
    ...                                     n_bootstrap=50)
    >>> model.fit(X, y_observed, event_indicator=delta)
    """

    def __init__(self, censoring='right', max_censor_iter=50, censor_tol=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.censoring = censoring
        self.max_censor_iter = max_censor_iter
        self.censor_tol = censor_tol

    def fit(self, X, y, event_indicator, weights=None, clusters=None):
        """
        Fit censored quantile regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Observed values (min of event time and censoring time for right-censoring).
        event_indicator : array-like of shape (n_samples,)
            1 = event observed (uncensored), 0 = censored.
        weights : array-like, optional
            Observation weights.
        clusters : array-like, optional
            Cluster labels for cluster-robust SEs.

        Returns
        -------
        self
        """
        X, y, weights = self._validate_inputs(X, y, weights)
        self._validate_tau()
        event = np.asarray(event_indicator).ravel()

        n_samples, n_features = X.shape
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        self._init_storage(n_features)
        self._X_aug = X_augmented
        self._y = y
        self._weights = weights

        # Initial fit on all data (treating all as uncensored)
        self._fit_coefficients(X_augmented, y, weights)

        # Iterative Powell algorithm
        for iteration in range(self.max_censor_iter):
            prev_coef = {}
            for q in self.tau:
                prev_coef[q] = {}
                for output in self.output_names_:
                    prev_coef[q][output] = np.concatenate(
                        ([self.intercept_[q][output]], self.coef_[q][output])
                    )

            # Construct working weights
            iter_weights = weights.copy()
            for q in self.tau:
                for ki, output in enumerate(self.output_names_):
                    coef_full = prev_coef[q][output]
                    predicted = X_augmented @ coef_full

                    for i in range(n_samples):
                        if event[i] == 0:  # censored
                            if self.censoring == 'right':
                                # Include if predicted quantile > censoring point
                                if predicted[i] <= y[i, ki]:
                                    iter_weights[i] = 0.0
                            else:  # left censoring
                                if predicted[i] >= y[i, ki]:
                                    iter_weights[i] = 0.0

            # Re-fit with adjusted weights
            nonzero = iter_weights > 0
            if np.sum(nonzero) < X_augmented.shape[1]:
                warnings.warn("Too few uncensored observations for estimation.")
                break

            self._fit_coefficients(X_augmented, y, iter_weights)

            # Check convergence
            max_change = 0
            for q in self.tau:
                for output in self.output_names_:
                    cur = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                    max_change = max(max_change, np.max(np.abs(cur - prev_coef[q][output])))

            if max_change < self.censor_tol:
                break

        # Compute pinball loss and standard errors on final model
        self._compute_pinball_losses(X_augmented, y, weights)

        if clusters is not None:
            self._compute_cluster_se(X_augmented, y, weights, np.asarray(clusters))
        elif self.se_method == 'bootstrap':
            self._compute_bootstrap_se(X_augmented, y, weights)
        elif self.se_method in ('analytical', 'kernel'):
            self._compute_analytical_se(X_augmented, y, method=self.se_method)

        for q in self.tau:
            for output in self.output_names_:
                self._is_fitted[q][output] = True

        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update({
            'censoring': self.censoring,
            'max_censor_iter': self.max_censor_iter,
            'censor_tol': self.censor_tol,
        })
        return params
