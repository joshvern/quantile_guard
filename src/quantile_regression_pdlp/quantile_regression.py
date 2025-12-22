# quantile_regression_pdlp/quantile_regression.py

from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from scipy.stats import t
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed
import multiprocessing
import threading


class QuantileRegression(BaseEstimator, RegressorMixin):
    """
    Quantile Regression using PDLP solver from Google's OR-Tools, with statistical summaries and multi-output support.

    Parameters
    ----------
    tau : float or list of floats, default=0.5
        The quantile(s) to estimate, each must be between 0 and 1.
        Can be a single float for one quantile or a list of floats for multiple quantiles.

    n_bootstrap : int, default=1000
        Number of bootstrap samples to use for estimating standard errors.

    random_state : int, default=None
        Seed for the random number generator.

    regularization : str, default='none'
        Type of regularization to apply. Options are 'l1' for Lasso regularization or 'none' for no regularization.

    alpha : float, default=0.0
        Regularization strength. Must be a non-negative float. Higher values imply stronger regularization.

    n_jobs : int, default=1
        The number of jobs to run in parallel for bootstrapping.
        `-1` means using all processors.

    Attributes
    ----------
    coef_ : dict
        Estimated coefficients for each quantile and output.
        Structure: {tau: {output: array of coefficients}}

    intercept_ : dict
        Estimated intercept term for each quantile and output.
        Structure: {tau: {output: float}}

    stderr_ : dict
        Standard errors of the coefficients for each quantile and output.
        Structure: {tau: {output: array of standard errors}}

    tvalues_ : dict
        T-statistics of the coefficients for each quantile and output.
        Structure: {tau: {output: array of t-values}}

    pvalues_ : dict
        P-values of the coefficients for each quantile and output.
        Structure: {tau: {output: array of p-values}}

    feature_names_ : list
        List of feature names. If input X is a pandas DataFrame, the column names are used; otherwise, generic names are assigned.

    output_names_ : list
        List of output names. If input y is a pandas DataFrame, the column names are used; otherwise, generic names are assigned.

    Methods
    -------
    fit(X, y, weights=None)
        Fit the quantile regression model.

    predict(X)
        Predict using the quantile regression model.

    summary()
        Return a summary of the regression results.
    """

    def __init__(self, tau=0.5, n_bootstrap=1000, random_state=None,
                 regularization='none', alpha=0.0, n_jobs=1):
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.regularization = regularization
        self.alpha = alpha
        self.n_jobs = n_jobs

        # Attributes initialized during fitting
        self.coef_ = None
        self.intercept_ = None
        self.stderr_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.feature_names_ = None
        self.output_names_ = None
        self._is_fitted = None

    def fit(self, X, y, weights=None):
        """
        Fit the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be a NumPy array or a pandas DataFrame.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values. Can be a NumPy array, pandas Series, or pandas DataFrame.

        weights : array-like of shape (n_samples,), optional
            Weights for each observation. Default is None, which assigns equal weight to all observations.

        Returns
        -------
        self : object
            Returns self.
        """
        # Handle pandas DataFrames and Series for X
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f'X{i}' for i in range(1, X.shape[1] + 1)]
            X = np.asarray(X)

        # Handle pandas DataFrames, Series, or NumPy arrays for y
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

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        # Handle weights
        if weights is None:
            weights = np.ones(n_samples)
        else:
            weights = np.asarray(weights)
            if weights.shape[0] != n_samples:
                raise ValueError("Weights array must have the same length as the number of observations.")

        # Validate tau
        self._validate_tau()

        # Add intercept term by appending a column of ones to X
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        # Initialize storage for multiple quantiles and outputs
        self.coef_ = {q: {output: np.zeros(n_features) for output in self.output_names_} for q in self.tau}
        self.intercept_ = {q: {output: 0.0 for output in self.output_names_} for q in self.tau}
        self.stderr_ = {q: {output: np.zeros(n_features + 1) for output in self.output_names_} for q in self.tau}
        self.tvalues_ = {q: {output: np.zeros(n_features + 1) for output in self.output_names_} for q in self.tau}
        self.pvalues_ = {q: {output: np.zeros(n_features + 1) for output in self.output_names_} for q in self.tau}
        self._is_fitted = {q: {output: False for output in self.output_names_} for q in self.tau}

        # Solve LP for all quantiles and outputs simultaneously with non-crossing constraints
        coefficients = self._solve_multiple_lp(X_augmented, y, weights)

        # Extract the coefficients
        for q in self.tau:
            for idx, output in enumerate(self.output_names_):
                self.intercept_[q][output] = coefficients[q][idx][0]
                self.coef_[q][output] = coefficients[q][idx][1:]

        # Compute standard errors via bootstrapping with progress indicators
        self._compute_standard_errors(X_augmented, y, weights)

        # Mark all quantiles and outputs as fitted
        for q in self.tau:
            for output in self.output_names_:
                self._is_fitted[q][output] = True

        return self

    def _validate_tau(self):
        """
        Validate the tau parameter to ensure all quantiles are between 0 and 1 and properly sorted.

        Raises
        ------
        ValueError, TypeError
        """
        if isinstance(self.tau, float):
            if not 0 < self.tau < 1:
                raise ValueError("Each quantile tau must be between 0 and 1.")
            self.tau = [self.tau]
        elif isinstance(self.tau, list):
            if not all(isinstance(q, float) and 0 < q < 1 for q in self.tau):
                raise ValueError("All quantiles tau must be floats between 0 and 1.")
            self.tau = sorted(self.tau)  # Sort to enforce ordering
        else:
            raise TypeError("tau must be a float or a list of floats.")

    def _solve_multiple_lp(self, X, y, weights, return_coefficients=True):
        """
        Solve multiple quantile regression problems as a single LP with non-crossing constraints.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented feature matrix with intercept.

        y : ndarray of shape (n_samples, n_outputs)
            Target values.

        weights : ndarray of shape (n_samples,)
            Weights for each observation.

        return_coefficients : bool, default=True
            Whether to return the estimated coefficients.

        Returns
        -------
        coefficients : dict (only if return_coefficients is True)
            Estimated coefficients for each quantile and output.
            Structure: {tau: {output: array of coefficients}}
        """
        n_samples, n_features_plus_1 = X.shape
        n_outputs = y.shape[1]
        n_quantiles = len(self.tau)

        # Create the solver instance
        solver = pywraplp.Solver.CreateSolver('PDLP')
        if not solver:
            raise Exception("PDLP solver is not available.")

        infinity = solver.infinity()

        # Define variables for each quantile and output
        # Structure: beta[q][k][j], r_pos[q][k][i], r_neg[q][k][i]
        beta = {q: {k: [solver.NumVar(-infinity, infinity, f'beta_{j}_q{q}_k{k}') 
                       for j in range(n_features_plus_1)] for k in range(n_outputs)} 
                for q in self.tau}
        r_pos = {q: {k: [solver.NumVar(0, infinity, f'r_pos_{i}_q{q}_k{k}') 
                         for i in range(n_samples)] for k in range(n_outputs)} 
                  for q in self.tau}
        r_neg = {q: {k: [solver.NumVar(0, infinity, f'r_neg_{i}_q{q}_k{k}') 
                         for i in range(n_samples)] for k in range(n_outputs)} 
                  for q in self.tau}

        # If L1 regularization is specified, introduce auxiliary variables for each quantile, output, and feature
        if self.regularization == 'l1' and self.alpha > 0:
            z = {q: {k: [solver.NumVar(0, infinity, f'z_{j}_q{q}_k{k}') 
                        for j in range(1, n_features_plus_1)] for k in range(n_outputs)} 
                 for q in self.tau}
            for q in self.tau:
                for k in range(n_outputs):
                    for j in range(1, n_features_plus_1):
                        # z_j_q_k >= beta_j_q_k
                        solver.Add(beta[q][k][j] <= z[q][k][j - 1])
                        # z_j_q_k >= -beta_j_q_k
                        solver.Add(-beta[q][k][j] <= z[q][k][j - 1])

        # Add constraints and objective
        objective = solver.Objective()
        for q in self.tau:
            for k in range(n_outputs):
                for i in range(n_samples):
                    # Residual constraints: y_i_k = x_i^T beta_q_k + r_pos_q_k_i - r_neg_q_k_i
                    constraint_expr = (
                        sum(X[i, j] * beta[q][k][j] for j in range(n_features_plus_1)) + 
                        r_pos[q][k][i] - 
                        r_neg[q][k][i]
                    )
                    solver.Add(constraint_expr == y[i, k])

                    # Objective coefficients
                    objective.SetCoefficient(r_pos[q][k][i], self.tau[self.tau.index(q)] * weights[i])
                    objective.SetCoefficient(r_neg[q][k][i], (1 - self.tau[self.tau.index(q)]) * weights[i])

        # Add L1 regularization to the objective if specified
        if self.regularization == 'l1' and self.alpha > 0:
            for q in self.tau:
                for k in range(n_outputs):
                    for j in range(n_features_plus_1 - 1):
                        objective.SetCoefficient(z[q][k][j], self.alpha)

        objective.SetMinimization()

        # Add non-crossing constraints per output
        for k in range(n_outputs):
            for i in range(n_samples):
                for q_idx in range(len(self.tau) - 1):
                    q_lower = self.tau[q_idx]
                    q_upper = self.tau[q_idx + 1]
                    # Predicted values for quantile q_lower <= predicted values for quantile q_upper
                    pred_lower = sum(X[i, j] * beta[q_lower][k][j] for j in range(n_features_plus_1))
                    pred_upper = sum(X[i, j] * beta[q_upper][k][j] for j in range(n_features_plus_1))
                    solver.Add(pred_lower <= pred_upper)

        # Solve the LP problem
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            raise Exception('Solver did not find an optimal solution.')

        if return_coefficients:
            # Extract the coefficients
            coefficients = {}
            for q in self.tau:
                coefficients[q] = {}
                for k in range(n_outputs):
                    intercept = beta[q][k][0].solution_value()
                    coef = np.array([beta[q][k][j].solution_value() for j in range(1, n_features_plus_1)])
                    coefficients[q][k] = np.concatenate(([intercept], coef))
            return coefficients

    def _compute_standard_errors(self, X, y, weights):
        """
        Compute standard errors via bootstrapping using parallel processing with progress indicators.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented feature matrix with intercept.

        y : ndarray of shape (n_samples, n_outputs)
            Target values.

        weights : ndarray of shape (n_samples,)
            Weights for each observation.
        """
        np.random.seed(self.random_state)
        n_samples, n_features_plus_1 = X.shape
        n_outputs = y.shape[1]
        n_quantiles = len(self.tau)

        # Initialize storage for bootstrap coefficients
        beta_bootstrap = {q: {output: np.zeros((self.n_bootstrap, n_features_plus_1)) 
                             for output in self.output_names_} for q in self.tau}

        # Determine the number of jobs
        if self.n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        # Create a shared counter using multiprocessing.Manager
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        # Define a lock for thread-safe counter updates
        lock = manager.Lock()

        # Function to perform a single bootstrap iteration
        def bootstrap_task(i):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            weights_sample = weights[sample_indices]
            try:
                # Solve for multiple quantiles and outputs
                beta_sample = self._solve_multiple_lp(X_sample, y_sample, weights_sample, return_coefficients=True)
                with lock:
                    counter.value += 1
                return beta_sample
            except Exception:
                with lock:
                    counter.value += 1
                return {q: {output: np.full(n_features_plus_1, np.nan) for output in self.output_names_} 
                        for q in self.tau}

        # Function to update the progress bar
        def update_pbar():
            with tqdm(total=self.n_bootstrap, desc='Bootstrapping') as pbar:
                previous = 0
                while True:
                    with lock:
                        current = counter.value
                    delta = current - previous
                    if delta > 0:
                        pbar.update(delta)
                        previous = current
                    if current >= self.n_bootstrap:
                        break

        # Start the progress bar updater thread
        thread = threading.Thread(target=update_pbar)
        thread.start()

        # Perform bootstrapping in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_task)(i) for i in range(self.n_bootstrap)
        )

        # Wait for the progress bar updater to finish
        thread.join()

        # Populate bootstrap coefficients
        for i, beta_sample in enumerate(results):
            for q in self.tau:
                for k, output in enumerate(self.output_names_):
                    beta_bootstrap[q][output][i, :] = beta_sample[q][k]

        # Compute standard errors, t-values, and p-values for each quantile and output
        for q in self.tau:
            for output in self.output_names_:
                # Remove any iterations where the solver failed
                valid_bootstrap = beta_bootstrap[q][output][~np.isnan(beta_bootstrap[q][output]).any(axis=1)]

                if valid_bootstrap.size == 0:
                    raise Exception(f"All bootstrap iterations failed for quantile {q} and output {output}.")

                # Compute standard errors
                stderr = np.std(valid_bootstrap, axis=0, ddof=1)
                self.stderr_[q][output] = stderr

                # Compute t-values and p-values
                coef_full = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                stderr_full = self.stderr_[q][output]
                tvalues_full = coef_full / stderr_full
                self.tvalues_[q][output] = tvalues_full

                df = len(y) - (X.shape[1] - 1)  # Degrees of freedom
                pvalues_full = 2 * (1 - t.cdf(np.abs(tvalues_full), df=df))
                self.pvalues_[q][output] = pvalues_full

    def predict(self, X):
        """
        Predict using the quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples. Can be a NumPy array or a pandas DataFrame.

        Returns
        -------
        y_pred : dict of dicts of ndarrays
            Predicted values for each quantile and output.
            Structure: {tau: {output: array of predictions}}
        """
        if not all(all(fit for fit in self._is_fitted[q].values()) for q in self.tau):
            raise Exception("Model is not fitted yet. Please call 'fit' before 'predict'.")

        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)

        n_samples = X.shape[0]

        # Add intercept term
        X_augmented = np.hstack((np.ones((n_samples, 1)), X))

        y_pred = {}
        for q in self.tau:
            y_pred[q] = {}
            for output in self.output_names_:
                coefficients = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                y_pred[q][output] = X_augmented @ coefficients
        return y_pred

    def summary(self):
        """
        Return a summary of the regression results.

        Returns
        -------
        summary_dict : dict of dicts of pandas DataFrames
            Summary tables for each quantile and output with coefficients, standard errors, t-values, and p-values.
            Structure: {tau: {output: DataFrame}}
        """
        if not all(all(fit for fit in self._is_fitted[q].values()) for q in self.tau):
            raise Exception("Model is not fitted yet. Please call 'fit' before 'summary'.")

        summary_dict = {}
        for q in self.tau:
            summary_dict[q] = {}
            for output in self.output_names_:
                coef = np.concatenate(([self.intercept_[q][output]], self.coef_[q][output]))
                index = ['Intercept'] + self.feature_names_
                summary_df = pd.DataFrame({
                    'Coefficient': coef,
                    'Std. Error': self.stderr_[q][output],
                    't-value': self.tvalues_[q][output],
                    'P>|t|': self.pvalues_[q][output],
                }, index=index)
                summary_dict[q][output] = summary_df
        return summary_dict

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'tau': self.tau,
            'n_bootstrap': self.n_bootstrap,
            'random_state': self.random_state,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
