"""Small fixed-sigma EM, adapted from Erwin's ``utils/custom_gmm.py``.

The original fixed-covariance update is retained; unused GPU/model-selection
branches are removed, and weighted observations avoid stochastic resampling.
"""

import math
import numpy as np


class FixedSigmaGMM:
    """Fit isotropic Gaussian centers with known standard deviation.

    This is a CPU NumPy estimator. ``fit`` accepts optional nonnegative sample
    weights, such as density times quadrature weights. Multiple seeded weighted
    k-means++ starts are compared by weighted log likelihood.
    """

    def __init__(self, n_components: int, sigma: float, *, n_init: int = 5,
                 max_iter: int = 100, tol: float = 1e-6, seed: int = 0):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if n_init < 1 or max_iter < 1 or tol <= 0:
            raise ValueError("n_init, max_iter, and tol must be positive")
        self.n_components, self.sigma = n_components, sigma
        self.n_init, self.max_iter, self.tol, self.seed = n_init, max_iter, tol, seed

    def fit(self, samples, sample_weight=None):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or len(samples) < self.n_components or not np.isfinite(samples).all():
            raise ValueError("samples must be finite [S,d] with S >= n_components")
        weight = np.ones(len(samples)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (len(samples),) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and have positive total")
        weight = weight / weight.sum()
        if np.count_nonzero(weight) < self.n_components:
            raise ValueError("fewer positive-weight samples than mixture components")
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            means = [samples[rng.choice(len(samples), p=weight)].copy()]
            distance = ((samples - means[0]) ** 2).sum(axis=1)
            for _ in range(1, self.n_components):
                probabilities = distance * weight
                if probabilities.sum() <= np.finfo(float).tiny:
                    raise ValueError("not enough distinct sample locations for this mixture")
                # Greedy local trials match the robust k-means++ initialization
                # used by the historical sklearn-backed implementation.
                candidates = rng.choice(len(samples), size=2 + int(np.log(self.n_components)),
                                        p=probabilities / probabilities.sum())
                candidate_distances = ((samples[:, None, :] - samples[candidates][None, :, :]) ** 2).sum(-1)
                updated = np.minimum(distance[:, None], candidate_distances)
                best_candidate = int((weight @ updated).argmin())
                means.append(samples[candidates[best_candidate]].copy())
                distance = updated[:, best_candidate]
            means = np.stack(means)
            mixture = np.full(self.n_components, 1 / self.n_components)
            previous = -np.inf
            converged = False
            for iteration in range(self.max_iter):
                log_prob = self._log_prob(samples, means, mixture)
                maximum = log_prob.max(axis=1, keepdims=True)
                log_sum = maximum[:, 0] + np.log(np.exp(log_prob - maximum).sum(axis=1))
                objective = float(weight @ log_sum)
                responsibility = np.exp(log_prob - log_sum[:, None]) * weight[:, None]
                masses = responsibility.sum(axis=0).clip(min=np.finfo(float).tiny)
                means = responsibility.T @ samples / masses[:, None]
                mixture = masses / masses.sum()
                if abs(objective - previous) < self.tol:
                    converged = True
                    break
                previous = objective
            if best is None or objective > best[0]:
                best = (objective, means.copy(), mixture.copy(), converged, iteration + 1)
        self.lower_bound_, self.means_, self.weights_, self.converged_, self.n_iter_ = best
        return self

    def _log_prob(self, samples, means, mixture):
        squared = ((samples[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
        return (-squared / (2 * self.sigma ** 2)
                - samples.shape[1] * math.log(self.sigma * math.sqrt(2 * math.pi))
                + np.log(mixture.clip(min=np.finfo(float).tiny))[None, :])

    def score_samples(self, samples):
        if not hasattr(self, "means_"):
            raise ValueError("fit the mixture before scoring samples")
        log_prob = self._log_prob(np.asarray(samples, dtype=np.float64), self.means_, self.weights_)
        maximum = log_prob.max(axis=1)
        return maximum + np.log(np.exp(log_prob - maximum[:, None]).sum(axis=1))
