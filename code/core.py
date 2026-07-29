from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import multivariate_normal, norm, qmc, wasserstein_distance


LOG_2PI = math.log(2.0 * math.pi)
TINY = np.finfo(np.float64).tiny


def stable_cholesky(
    matrix: np.ndarray, jitter: float = 1e-10, max_tries: int = 12
) -> tuple[np.ndarray, float]:
    """Return a lower Cholesky factor and the diagonal jitter used.

    The initial jitter is scale-aware.  This matters for the Woodbury middle
    matrix, whose entries can be O(1e16) when an optimizer probes a very small
    diagonal variance: an absolute O(1e-12) perturbation is then rounded away
    even though the matrix is positive definite in exact arithmetic.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    if not np.all(np.isfinite(matrix)):
        raise np.linalg.LinAlgError(
            "non-finite matrix passed to stable_cholesky"
        )
    eye = np.eye(matrix.shape[0], dtype=np.float64)
    matrix_scale = max(
        1.0, float(np.max(np.abs(np.diag(matrix))))
    )
    current = max(
        float(jitter),
        10.0 * np.finfo(np.float64).eps * matrix_scale,
    )
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(matrix + current * eye), current
        except np.linalg.LinAlgError:
            current *= 10.0
    return np.linalg.cholesky(matrix + current * eye), current


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def positive_map_forward(
    x: np.ndarray, name: str = "softplus", scale: float = 1.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if name == "softplus":
        return softplus(x)
    if name == "exp":
        return np.exp(np.clip(x, -745.0, 700.0))
    if name == "squareplus":
        if scale <= 0.0:
            raise ValueError("squareplus scale must be positive")
        root = np.sqrt(x * x + scale * scale)
        output = np.empty_like(x)
        positive = x >= 0.0
        output[positive] = 0.5 * (x[positive] + root[positive])
        output[~positive] = scale * scale / (
            2.0 * (root[~positive] - x[~positive])
        )
        return output
    raise ValueError(f"unknown positive map: {name}")


def positive_map_inverse(
    y: np.ndarray, name: str = "softplus", scale: float = 1.0
) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=np.float64), TINY)
    if name == "softplus":
        return y + np.log(-np.expm1(-y))
    if name == "exp":
        return np.log(y)
    if name == "squareplus":
        if scale <= 0.0:
            raise ValueError("squareplus scale must be positive")
        return y - scale * scale / (4.0 * y)
    raise ValueError(f"unknown positive map: {name}")


def positive_map_derivative(
    x: np.ndarray, name: str = "softplus", scale: float = 1.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if name == "softplus":
        return sigmoid(x)
    if name == "exp":
        return np.exp(np.clip(x, -745.0, 700.0))
    if name == "squareplus":
        root = np.sqrt(x * x + scale * scale)
        return positive_map_forward(x, name, scale) / root
    raise ValueError(f"unknown positive map: {name}")


def positive_map_log_jacobian(
    x: np.ndarray, name: str = "softplus", scale: float = 1.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if name == "softplus":
        return -np.logaddexp(0.0, -x)
    if name == "exp":
        return x
    if name == "squareplus":
        root = np.sqrt(x * x + scale * scale)
        return np.log(positive_map_forward(x, name, scale)) - np.log(root)
    raise ValueError(f"unknown positive map: {name}")


def positive_map_log_jacobian_derivative(
    x: np.ndarray, name: str = "softplus", scale: float = 1.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if name == "softplus":
        return 1.0 - sigmoid(x)
    if name == "exp":
        return np.ones_like(x)
    if name == "squareplus":
        root = np.sqrt(x * x + scale * scale)
        return (root - x) / (root * root)
    raise ValueError(f"unknown positive map: {name}")


def true_function(name: str, x: np.ndarray) -> np.ndarray:
    """The three one-dimensional functions used in the paper Table 5."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if name == "linear":
        return values
    if name == "log":
        return 2.0 * np.log1p(values)
    if name == "sigmoid":
        return 3.0 / (1.0 + np.exp(-4.0 * values + 5.0))
    raise ValueError(f"unknown benchmark: {name}")


def make_1d_case(
    name: str, seed: int, n_train: int = 30, noise_std: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n_train, dtype=np.float64)[:, None]
    y = true_function(name, x) + noise_std * rng.standard_normal(n_train)
    return x, y


def rbf_kernel(
    x: np.ndarray,
    z: np.ndarray,
    signal_variance: float,
    lengthscale: float,
) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    difference = x[:, None, :] - z[None, :, :]
    squared_distance = np.sum(difference * difference, axis=2)
    return signal_variance * np.exp(
        -0.5 * squared_distance / (lengthscale * lengthscale)
    )


def covariance_function_derivative(
    x: np.ndarray,
    z: np.ndarray,
    derivative_dimensions: np.ndarray,
    signal_variance: float,
    lengthscale: float,
) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    dimensions = np.asarray(derivative_dimensions, dtype=int)
    kernel = rbf_kernel(x, z, signal_variance, lengthscale)
    difference = x[:, None, :] - z[None, :, :]
    selected = difference[:, np.arange(len(z)), dimensions]
    return kernel * selected / (lengthscale * lengthscale)


def covariance_derivative_derivative(
    z1: np.ndarray,
    dimensions1: np.ndarray,
    z2: np.ndarray,
    dimensions2: np.ndarray,
    signal_variance: float,
    lengthscale: float,
) -> np.ndarray:
    z1 = np.atleast_2d(np.asarray(z1, dtype=np.float64))
    z2 = np.atleast_2d(np.asarray(z2, dtype=np.float64))
    dimensions1 = np.asarray(dimensions1, dtype=int)
    dimensions2 = np.asarray(dimensions2, dtype=int)
    kernel = rbf_kernel(z1, z2, signal_variance, lengthscale)
    difference = z1[:, None, :] - z2[None, :, :]
    output = np.empty((len(z1), len(z2)), dtype=np.float64)
    l2 = lengthscale * lengthscale
    l4 = l2 * l2
    for i, first in enumerate(dimensions1):
        for j, second in enumerate(dimensions2):
            kronecker = 1.0 if first == second else 0.0
            output[i, j] = kernel[i, j] * (
                kronecker / l2
                - difference[i, j, first] * difference[i, j, second] / l4
            )
    return output


def optimize_gp_hyperparameters(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[float, float, float]:
    """Match the submitted paper protocol exactly."""
    centered = y_train - float(np.mean(y_train))
    initial = np.log(
        [max(float(np.var(centered)), 0.2), 2.0, 0.35]
    )

    def objective(log_parameters: np.ndarray) -> float:
        signal_variance, lengthscale, noise_std = np.exp(log_parameters)
        kernel = rbf_kernel(
            x_train, x_train, signal_variance, lengthscale
        ) + (noise_std * noise_std + 1e-7) * np.eye(len(x_train))
        try:
            factor = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve(factor, centered, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e25
        logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
        return float(
            0.5 * centered @ alpha
            + 0.5 * logdet
            + 0.5 * len(x_train) * LOG_2PI
        )

    bounds = [
        (math.log(1e-3), math.log(200.0)),
        (math.log(0.15), math.log(30.0)),
        (math.log(0.03), math.log(3.0)),
    ]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 80},
    )
    return tuple(float(value) for value in np.exp(result.x))


@dataclass
class GaussianTarget:
    mean: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    factor: tuple[np.ndarray, bool]
    logdet: float
    jitter: float

    @classmethod
    def from_moments(
        cls, mean: np.ndarray, covariance: np.ndarray
    ) -> "GaussianTarget":
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        covariance = np.asarray(covariance, dtype=np.float64)
        cholesky, jitter = stable_cholesky(covariance)
        stabilized = (
            0.5 * (covariance + covariance.T)
            + jitter * np.eye(len(mean), dtype=np.float64)
        )
        factor = cho_factor(stabilized, lower=True, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        return cls(mean, stabilized, cholesky, factor, logdet, jitter)

    @property
    def dimension(self) -> int:
        return len(self.mean)

    def logpdf(self, values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        difference = values - self.mean[None, :]
        solved = cho_solve(
            self.factor, difference.T, check_finite=False
        ).T
        quadratic = np.sum(difference * solved, axis=1)
        return -0.5 * (
            self.dimension * LOG_2PI + self.logdet + quadratic
        )

    def score(self, values: np.ndarray) -> np.ndarray:
        difference = (
            np.atleast_2d(np.asarray(values, dtype=np.float64))
            - self.mean[None, :]
        )
        return -cho_solve(
            self.factor, difference.T, check_finite=False
        ).T


def make_derivative_target(
    benchmark: str,
    data_seed: int,
    m_constraints: int,
    n_train: int = 30,
) -> tuple[GaussianTarget, dict[str, Any]]:
    """Construct the exact posterior derivative marginal used by all methods."""
    x_train, y_train = make_1d_case(
        benchmark, data_seed, n_train=n_train, noise_std=0.2
    )
    signal_variance, lengthscale, noise_std = optimize_gp_hyperparameters(
        x_train, y_train
    )
    centered = y_train - float(np.mean(y_train))
    training_covariance = rbf_kernel(
        x_train, x_train, signal_variance, lengthscale
    ) + (noise_std * noise_std + 1e-7) * np.eye(n_train)
    training_factor = cho_factor(
        training_covariance, lower=True, check_finite=False
    )
    alpha = cho_solve(training_factor, centered, check_finite=False)
    constraints = np.linspace(
        1.0, 9.0, m_constraints, dtype=np.float64
    )[:, None]
    derivative_dimensions = np.zeros(m_constraints, dtype=int)
    cross = covariance_function_derivative(
        x_train,
        constraints,
        derivative_dimensions,
        signal_variance,
        lengthscale,
    ).T
    prior = covariance_derivative_derivative(
        constraints,
        derivative_dimensions,
        constraints,
        derivative_dimensions,
        signal_variance,
        lengthscale,
    )
    mean = cross @ alpha
    covariance = prior - cross @ cho_solve(
        training_factor, cross.T, check_finite=False
    )
    covariance = 0.5 * (covariance + covariance.T)
    target = GaussianTarget.from_moments(mean, covariance)
    metadata = {
        "benchmark": benchmark,
        "data_seed": int(data_seed),
        "n_train": int(n_train),
        "training_noise_generation_std": 0.2,
        "m_constraints": int(m_constraints),
        "constraints": constraints[:, 0].tolist(),
        "signal_variance": signal_variance,
        "lengthscale": lengthscale,
        "fitted_noise_std": noise_std,
        "target_jitter": target.jitter,
    }
    return target, metadata


def basis_derivative_point(
    benchmark: str, data_seed: int, constraints: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Paper deterministic constrained-basis baseline at derivative locations."""
    x_train, y_train = make_1d_case(benchmark, data_seed)
    x = x_train[:, 0]
    eps = 1e-4

    def phi(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.ones_like(values),
                values,
                np.log1p(values),
                np.sqrt(values + eps),
            ]
        )

    def derivative_phi(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.zeros_like(values),
                np.ones_like(values),
                1.0 / (1.0 + values),
                0.5 / np.sqrt(values + eps),
            ]
        )

    design = phi(x)
    dense = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    derivative_design = derivative_phi(dense)
    result = minimize(
        lambda weights: float(np.sum((design @ weights - y_train) ** 2)),
        np.zeros(design.shape[1], dtype=np.float64),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda weights: derivative_design @ weights,
            }
        ],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    derivative = derivative_phi(
        np.asarray(constraints, dtype=np.float64).reshape(-1)
    ) @ result.x
    derivative = np.maximum(derivative, 0.0)
    return derivative, {
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
    }


def ep_probit_moments(
    target: GaussianTarget,
    softness: float = 0.35,
    damping: float = 0.7,
    max_sweeps: int = 100,
    tolerance: float = 1e-8,
) -> tuple[GaussianTarget, dict[str, Any]]:
    """Gaussian-site EP for prod_j Phi(d_j / softness)."""
    dimension = target.dimension
    prior_precision = cho_solve(
        target.factor, np.eye(dimension), check_finite=False
    )
    prior_natural = prior_precision @ target.mean
    site_precision = np.zeros(dimension, dtype=np.float64)
    site_natural = np.zeros(dimension, dtype=np.float64)
    posterior_mean = target.mean.copy()
    posterior_covariance = target.covariance.copy()
    converged = False
    max_change = float("inf")
    failure_reason = ""
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        max_change = 0.0
        for coordinate in range(dimension):
            marginal_variance = float(
                posterior_covariance[coordinate, coordinate]
            )
            marginal_mean = float(posterior_mean[coordinate])
            cavity_precision = (
                1.0 / marginal_variance - site_precision[coordinate]
            )
            if cavity_precision <= 1e-12:
                failure_reason = f"invalid cavity precision at {coordinate}"
                break
            cavity_variance = 1.0 / cavity_precision
            cavity_natural = (
                marginal_mean / marginal_variance
                - site_natural[coordinate]
            )
            cavity_mean = cavity_variance * cavity_natural
            scale = math.sqrt(cavity_variance + softness * softness)
            standardized = cavity_mean / scale
            log_ratio = (
                -0.5 * standardized * standardized
                - 0.5 * LOG_2PI
                - float(log_ndtr(standardized))
            )
            ratio = math.exp(min(log_ratio, 700.0))
            tilted_mean = (
                cavity_mean + cavity_variance / scale * ratio
            )
            tilted_variance = cavity_variance - (
                cavity_variance
                * cavity_variance
                / (cavity_variance + softness * softness)
                * ratio
                * (ratio + standardized)
            )
            if tilted_variance <= 1e-12:
                failure_reason = f"invalid tilted variance at {coordinate}"
                break
            proposed_precision = max(
                1.0 / tilted_variance - cavity_precision, 0.0
            )
            proposed_natural = (
                tilted_mean / tilted_variance - cavity_natural
            )
            updated_precision = (
                (1.0 - damping) * site_precision[coordinate]
                + damping * proposed_precision
            )
            updated_natural = (
                (1.0 - damping) * site_natural[coordinate]
                + damping * proposed_natural
            )
            max_change = max(
                max_change,
                abs(updated_precision - site_precision[coordinate]),
                abs(updated_natural - site_natural[coordinate]),
            )
            site_precision[coordinate] = updated_precision
            site_natural[coordinate] = updated_natural
            posterior_precision = prior_precision + np.diag(site_precision)
            cholesky, _ = stable_cholesky(
                posterior_precision, jitter=1e-12
            )
            posterior_covariance = cho_solve(
                (cholesky, True),
                np.eye(dimension),
                check_finite=False,
            )
            posterior_covariance = 0.5 * (
                posterior_covariance + posterior_covariance.T
            )
            posterior_mean = posterior_covariance @ (
                prior_natural + site_natural
            )
        if failure_reason:
            break
        if max_change < tolerance:
            converged = True
            break
    if failure_reason:
        raise RuntimeError(failure_reason)
    return GaussianTarget.from_moments(
        posterior_mean, posterior_covariance
    ), {
        "converged": converged,
        "sweeps": sweep,
        "max_site_change": max_change,
        "softness": softness,
    }


def family_rank(family: str) -> int:
    if family == "diag":
        return 0
    if family == "full":
        return -1
    if family.startswith("rank"):
        return int(family[4:])
    raise ValueError(f"unknown family: {family}")


def lower_indices(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(dimension)


def unpack_low_rank(
    parameters: np.ndarray, dimension: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = parameters[:dimension]
    log_diagonal = parameters[dimension : 2 * dimension]
    low_rank = (
        parameters[2 * dimension :].reshape(dimension, rank)
        if rank
        else np.zeros((dimension, 0), dtype=np.float64)
    )
    return mean, log_diagonal, low_rank


def unpack_full(
    parameters: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = parameters[:dimension]
    rows, columns = lower_indices(dimension)
    raw = parameters[dimension:]
    lower = np.zeros((dimension, dimension), dtype=np.float64)
    lower[rows, columns] = raw
    diagonal = np.diag_indices(dimension)
    lower[diagonal] = np.exp(lower[diagonal])
    return mean, lower


def low_rank_logdet_inverse_terms(
    log_diagonal: np.ndarray, low_rank: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Woodbury logdet, inverse diagonal, and inverse times low-rank factor."""
    inverse_diagonal_variance = np.exp(-2.0 * log_diagonal)
    rank = low_rank.shape[1]
    if rank == 0:
        return (
            float(2.0 * np.sum(log_diagonal)),
            inverse_diagonal_variance,
            np.zeros_like(low_rank),
        )
    scaled = inverse_diagonal_variance[:, None] * low_rank
    middle = np.eye(rank) + low_rank.T @ scaled
    middle_cholesky, _ = stable_cholesky(middle, jitter=1e-12)
    solved = cho_solve(
        (middle_cholesky, True), scaled.T, check_finite=False
    ).T
    inverse_diagonal = inverse_diagonal_variance - np.sum(
        scaled * solved, axis=1
    )
    logdet_middle = 2.0 * float(
        np.sum(np.log(np.diag(middle_cholesky)))
    )
    logdet = float(2.0 * np.sum(log_diagonal) + logdet_middle)
    return logdet, inverse_diagonal, solved


def dense_low_rank_logdet_inverse_terms(
    log_diagonal: np.ndarray, low_rank: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Dense reference used only by equivalence tests."""
    diagonal = np.exp(log_diagonal)
    covariance = np.diag(diagonal * diagonal) + low_rank @ low_rank.T
    cholesky, jitter = stable_cholesky(covariance)
    inverse = cho_solve(
        (cholesky, True), np.eye(len(diagonal)), check_finite=False
    )
    if jitter:
        covariance = covariance + jitter * np.eye(len(diagonal))
    logdet = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return logdet, np.diag(inverse), inverse @ low_rank


@dataclass
class TransportFit:
    family: str
    rank: int
    map_name: str
    map_scale: float
    parameters: np.ndarray
    validation_elbo: float
    normalized_gradient: float
    adam_steps: int
    lbfgs_iterations: int
    wall_time_seconds: float
    plateau_passed: bool
    gradient_passed: bool
    independent_elbo_1: float
    independent_elbo_1_se: float
    independent_elbo_2: float
    independent_elbo_2_se: float
    independent_agreement_passed: bool
    converged: bool
    failure_reason: str
    history: list[dict[str, float]]
    initialization: str = "independent"
    lbfgs_function_evaluations: int = 0

    def jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameters"] = self.parameters.tolist()
        return result


def initial_transport_parameters(
    target: GaussianTarget,
    family: str,
    seed: int,
    map_name: str,
    map_scale: float,
) -> np.ndarray:
    dimension = target.dimension
    marginal_sd = np.sqrt(
        np.clip(np.diag(target.covariance), 1e-12, None)
    )
    initial_derivative = np.maximum(
        target.mean, 0.20 * marginal_sd
    )
    auxiliary_mean = np.clip(
        positive_map_inverse(
            initial_derivative, map_name, map_scale
        ),
        -15.0,
        15.0,
    )
    jacobian = np.clip(
        positive_map_derivative(
            auxiliary_mean, map_name, map_scale
        ),
        1e-4,
        None,
    )
    delta_covariance = (
        0.55
        * target.covariance
        / (jacobian[:, None] * jacobian[None, :])
    )
    delta_covariance = 0.5 * (
        delta_covariance + delta_covariance.T
    )
    rank = family_rank(family)
    rng = np.random.default_rng(seed)
    if rank >= 0:
        if rank:
            eigenvalues, eigenvectors = np.linalg.eigh(
                delta_covariance
            )
            retained_rank = min(rank, dimension)
            order = np.argsort(eigenvalues)[::-1][
                :retained_rank
            ]
            retained = np.maximum(eigenvalues[order], 0.0)
            low_rank = np.zeros(
                (dimension, rank), dtype=np.float64
            )
            low_rank[:, :retained_rank] = (
                eigenvectors[:, order]
                * np.sqrt(retained)[None, :]
            )
            # Restart-specific perturbations break exact symmetries without
            # consulting hard-reference samples.
            low_rank += (
                1e-4
                * max(float(np.median(marginal_sd)), 1e-6)
                * rng.standard_normal(low_rank.shape)
            )
            residual_diagonal = np.diag(
                delta_covariance - low_rank @ low_rank.T
            )
        else:
            low_rank = np.zeros(
                (dimension, 0), dtype=np.float64
            )
            residual_diagonal = np.diag(delta_covariance)
        auxiliary_sd = np.sqrt(
            np.clip(residual_diagonal, 1e-10, 9.0)
        )
        return np.concatenate(
            [auxiliary_mean, np.log(auxiliary_sd), low_rank.ravel()]
        )
    transformed_covariance = delta_covariance.copy()
    # Bound the initialization scale by multiplying the *whole* covariance.
    # Clipping only its diagonal can destroy positive definiteness when the
    # delta-method Jacobian is tiny and off-diagonal entries are large.
    largest_diagonal = max(
        float(np.max(np.diag(transformed_covariance))), 1e-14
    )
    if largest_diagonal > 9.0:
        transformed_covariance *= 9.0 / largest_diagonal
    eigenvalues = np.linalg.eigvalsh(transformed_covariance)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum_eigenvalue = float(np.min(eigenvalues))
    spectral_floor = 100.0 * np.finfo(np.float64).eps * spectral_scale
    if minimum_eigenvalue < spectral_floor:
        transformed_covariance += (
            spectral_floor - minimum_eigenvalue
        ) * np.eye(dimension)
    lower, _ = stable_cholesky(
        transformed_covariance, jitter=1e-12
    )
    rows, columns = lower_indices(dimension)
    raw = lower[rows, columns].copy()
    diagonal_positions = rows == columns
    raw[diagonal_positions] = np.log(
        np.clip(raw[diagonal_positions], 1e-7, 3.0)
    )
    return np.concatenate([auxiliary_mean, raw])


def transport_parameter_count(dimension: int, family: str) -> int:
    rank = family_rank(family)
    if rank >= 0:
        return 2 * dimension + dimension * rank
    return dimension + dimension * (dimension + 1) // 2


def transport_covariance(
    parameters: np.ndarray, dimension: int, family: str
) -> np.ndarray:
    rank = family_rank(family)
    if rank >= 0:
        _, log_diagonal, low_rank = unpack_low_rank(
            parameters, dimension, rank
        )
        diagonal = np.exp(log_diagonal)
        covariance = np.diag(diagonal * diagonal)
        if rank:
            covariance += low_rank @ low_rank.T
        return 0.5 * (covariance + covariance.T)
    _, lower = unpack_full(parameters, dimension)
    return lower @ lower.T


def draw_auxiliary(
    parameters: np.ndarray,
    dimension: int,
    family: str,
    diagonal_noise: np.ndarray,
    rank_noise: np.ndarray | None,
) -> np.ndarray:
    rank = family_rank(family)
    if rank >= 0:
        mean, log_diagonal, low_rank = unpack_low_rank(
            parameters, dimension, rank
        )
        output = (
            mean[None, :]
            + diagonal_noise * np.exp(log_diagonal)[None, :]
        )
        if rank:
            if rank_noise is None:
                raise ValueError("rank noise is required")
            output += rank_noise @ low_rank.T
        return output
    mean, lower = unpack_full(parameters, dimension)
    return mean[None, :] + diagonal_noise @ lower.T


def transport_logpdf_auxiliary(
    parameters: np.ndarray,
    dimension: int,
    family: str,
    values: np.ndarray,
) -> np.ndarray:
    rank = family_rank(family)
    if rank >= 0:
        mean, log_diagonal, low_rank = unpack_low_rank(
            parameters, dimension, rank
        )
        difference = np.atleast_2d(values) - mean[None, :]
        inverse_diagonal_variance = np.exp(
            -2.0 * log_diagonal
        )
        base_quadratic = np.sum(
            difference
            * difference
            * inverse_diagonal_variance[None, :],
            axis=1,
        )
        logdet, _, _ = low_rank_logdet_inverse_terms(
            log_diagonal, low_rank
        )
        if rank:
            scaled = (
                inverse_diagonal_variance[:, None] * low_rank
            )
            middle = np.eye(rank) + low_rank.T @ scaled
            middle_cholesky, _ = stable_cholesky(
                middle, jitter=1e-12
            )
            projected = difference @ scaled
            correction = np.sum(
                projected
                * cho_solve(
                    (middle_cholesky, True),
                    projected.T,
                    check_finite=False,
                ).T,
                axis=1,
            )
            quadratic = base_quadratic - correction
        else:
            quadratic = base_quadratic
        return -0.5 * (
            dimension * LOG_2PI + logdet + quadratic
        )
    mean, lower = unpack_full(parameters, dimension)
    difference = np.atleast_2d(values) - mean[None, :]
    solved = np.linalg.solve(lower, difference.T).T
    logdet = 2.0 * float(np.sum(np.log(np.diag(lower))))
    return -0.5 * (
        dimension * LOG_2PI
        + logdet
        + np.sum(solved * solved, axis=1)
    )


def transport_logpdf_derivative(
    fit: TransportFit, derivatives: np.ndarray
) -> np.ndarray:
    derivatives = np.atleast_2d(
        np.asarray(derivatives, dtype=np.float64)
    )
    inside = np.all(derivatives > 0.0, axis=1)
    output = np.full(len(derivatives), -np.inf, dtype=np.float64)
    if np.any(inside):
        auxiliary = positive_map_inverse(
            derivatives[inside], fit.map_name, fit.map_scale
        )
        output[inside] = transport_logpdf_auxiliary(
            fit.parameters,
            derivatives.shape[1],
            fit.family,
            auxiliary,
        ) - np.sum(
            positive_map_log_jacobian(
                auxiliary, fit.map_name, fit.map_scale
            ),
            axis=1,
        )
    return output


def elbo_and_gradient(
    parameters: np.ndarray,
    target: GaussianTarget,
    family: str,
    diagonal_noise: np.ndarray,
    rank_noise: np.ndarray | None,
    map_name: str,
    map_scale: float,
    use_dense_low_rank_entropy: bool = False,
) -> tuple[float, np.ndarray]:
    """Pathwise ELBO and analytic gradient.

    For diag-plus-low-rank families, entropy and its gradient use Woodbury
    identities.  ``use_dense_low_rank_entropy`` exists solely for tests.
    """
    dimension = target.dimension
    rank = family_rank(family)
    auxiliary = draw_auxiliary(
        parameters,
        dimension,
        family,
        diagonal_noise,
        rank_noise,
    )
    derivatives = positive_map_forward(
        auxiliary, map_name, map_scale
    )
    jacobian = positive_map_derivative(
        auxiliary, map_name, map_scale
    )
    log_unnormalized = target.logpdf(derivatives) + np.sum(
        positive_map_log_jacobian(
            auxiliary, map_name, map_scale
        ),
        axis=1,
    )
    auxiliary_score = (
        jacobian * target.score(derivatives)
        + positive_map_log_jacobian_derivative(
            auxiliary, map_name, map_scale
        )
    )
    if rank >= 0:
        _, log_diagonal, low_rank = unpack_low_rank(
            parameters, dimension, rank
        )
        diagonal = np.exp(log_diagonal)
        if use_dense_low_rank_entropy:
            logdet, inverse_diagonal, inverse_low_rank = (
                dense_low_rank_logdet_inverse_terms(
                    log_diagonal, low_rank
                )
            )
        else:
            logdet, inverse_diagonal, inverse_low_rank = (
                low_rank_logdet_inverse_terms(
                    log_diagonal, low_rank
                )
            )
        entropy = 0.5 * (
            dimension * (1.0 + LOG_2PI) + logdet
        )
        gradient_mean = np.mean(auxiliary_score, axis=0)
        gradient_log_diagonal = np.mean(
            auxiliary_score
            * diagonal_noise
            * diagonal[None, :],
            axis=0,
        ) + inverse_diagonal * diagonal * diagonal
        if rank:
            if rank_noise is None:
                raise ValueError("rank noise is required")
            gradient_low_rank = (
                auxiliary_score.T @ rank_noise / len(auxiliary)
                + inverse_low_rank
            )
            gradient = np.concatenate(
                [
                    gradient_mean,
                    gradient_log_diagonal,
                    gradient_low_rank.ravel(),
                ]
            )
        else:
            gradient = np.concatenate(
                [gradient_mean, gradient_log_diagonal]
            )
    else:
        _, lower = unpack_full(parameters, dimension)
        entropy = (
            0.5 * dimension * (1.0 + LOG_2PI)
            + float(np.sum(np.log(np.diag(lower))))
        )
        gradient_mean = np.mean(auxiliary_score, axis=0)
        gradient_lower = (
            auxiliary_score.T @ diagonal_noise / len(auxiliary)
        )
        rows, columns = lower_indices(dimension)
        gradient_raw = gradient_lower[rows, columns].copy()
        diagonal_positions = rows == columns
        gradient_raw[diagonal_positions] *= lower[
            rows[diagonal_positions], columns[diagonal_positions]
        ]
        gradient_raw[diagonal_positions] += 1.0
        gradient = np.concatenate(
            [gradient_mean, gradient_raw]
        )
    return (
        float(np.mean(log_unnormalized) + entropy),
        gradient,
    )


def project_transport_parameters(
    parameters: np.ndarray, dimension: int, family: str
) -> np.ndarray:
    output = np.asarray(parameters, dtype=np.float64).copy()
    output[:dimension] = np.clip(
        output[:dimension], -25.0, 25.0
    )
    rank = family_rank(family)
    if rank >= 0:
        output[dimension : 2 * dimension] = np.clip(
            output[dimension : 2 * dimension], -16.0, 3.0
        )
        if rank:
            output[2 * dimension :] = np.clip(
                output[2 * dimension :], -5.0, 5.0
            )
    else:
        rows, columns = lower_indices(dimension)
        raw = output[dimension:]
        diagonal_positions = rows == columns
        raw[diagonal_positions] = np.clip(
            raw[diagonal_positions], -16.0, 3.0
        )
        raw[~diagonal_positions] = np.clip(
            raw[~diagonal_positions], -5.0, 5.0
        )
        output[dimension:] = raw
    return output


def transport_parameter_bounds(
    dimension: int, family: str
) -> list[tuple[float, float]]:
    rank = family_rank(family)
    if rank >= 0:
        return (
            [(-25.0, 25.0)] * dimension
            + [(-16.0, 3.0)] * dimension
            + [(-5.0, 5.0)] * (dimension * rank)
        )
    rows, columns = lower_indices(dimension)
    lower_bounds = [
        (-16.0, 3.0) if row == column else (-5.0, 5.0)
        for row, column in zip(rows, columns)
    ]
    return [(-25.0, 25.0)] * dimension + lower_bounds


def transport_optimization_scales(
    target: GaussianTarget,
    family: str,
    map_name: str,
    map_scale: float,
) -> np.ndarray:
    """Dimension-aware L-BFGS preconditioner in physical parameter space."""
    dimension = target.dimension
    marginal_sd = np.sqrt(
        np.clip(np.diag(target.covariance), 1e-14, None)
    )
    reference_derivative = np.maximum(
        target.mean, 0.2 * marginal_sd
    )
    reference_auxiliary = positive_map_inverse(
        reference_derivative, map_name, map_scale
    )
    jacobian = np.clip(
        positive_map_derivative(
            reference_auxiliary, map_name, map_scale
        ),
        1e-8,
        None,
    )
    auxiliary_scale = np.clip(
        marginal_sd / jacobian, 1e-5, 1.0
    )
    rank = family_rank(family)
    if rank >= 0:
        return np.concatenate(
            [
                auxiliary_scale,
                np.ones(dimension, dtype=np.float64),
                np.repeat(auxiliary_scale, rank),
            ]
        )
    rows, columns = lower_indices(dimension)
    raw_scales = np.where(
        rows == columns, 1.0, auxiliary_scale[rows]
    )
    return np.concatenate([auxiliary_scale, raw_scales])


def sobol_standard_normal(
    count: int, dimension: int, seed: int
) -> np.ndarray:
    """Scrambled Sobol normals with exact first/second-moment matching.

    Moment matching is a deterministic randomized-QMC control variate.  It is
    crucial for very concentrated GP derivative posteriors, where a tiny
    residual sample-mean error is amplified by the target precision.
    """
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    power = int(math.ceil(math.log2(max(count, 2))))
    uniforms = engine.random_base2(power)[:count]
    uniforms = np.clip(
        uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    normal = ndtri(uniforms)
    normal -= np.mean(normal, axis=0, keepdims=True)
    empirical = normal.T @ normal / len(normal)
    cholesky, _ = stable_cholesky(empirical, jitter=1e-14)
    normal = np.linalg.solve(cholesky, normal.T).T
    return normal


def split_noise(
    standard_normal: np.ndarray,
    dimension: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    diagonal = standard_normal[:, :dimension]
    rank_noise = (
        standard_normal[:, dimension : dimension + rank]
        if rank > 0
        else None
    )
    return diagonal, rank_noise


def independent_elbo_estimate(
    fit: TransportFit,
    target: GaussianTarget,
    count: int,
    seed: int,
    chunk_size: int = 10_000,
) -> tuple[float, float]:
    """Independent per-sample ELBO mean and Monte Carlo standard error."""
    rng = np.random.default_rng(seed)
    dimension = target.dimension
    total = 0.0
    total_square = 0.0
    observed = 0
    for start in range(0, count, chunk_size):
        current = min(chunk_size, count - start)
        diagonal_noise = rng.standard_normal(
            (current, dimension)
        )
        rank_noise = (
            rng.standard_normal((current, fit.rank))
            if fit.rank > 0
            else None
        )
        auxiliary = draw_auxiliary(
            fit.parameters,
            dimension,
            fit.family,
            diagonal_noise,
            rank_noise,
        )
        derivatives = positive_map_forward(
            auxiliary, fit.map_name, fit.map_scale
        )
        terms = (
            target.logpdf(derivatives)
            + np.sum(
                positive_map_log_jacobian(
                    auxiliary, fit.map_name, fit.map_scale
                ),
                axis=1,
            )
            - transport_logpdf_auxiliary(
                fit.parameters,
                dimension,
                fit.family,
                auxiliary,
            )
        )
        total += float(np.sum(terms))
        total_square += float(np.sum(terms * terms))
        observed += current
    mean = total / observed
    variance = max(
        (total_square - observed * mean * mean)
        / max(observed - 1, 1),
        0.0,
    )
    return float(mean), float(math.sqrt(variance / observed))


def optimize_transport(
    target: GaussianTarget,
    family: str,
    seed: int,
    map_name: str = "softplus",
    map_scale: float = 1.0,
    max_steps: int = 8000,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    check_every: int = 200,
    validation_samples: int = 8192,
    polish_samples: int = 8192,
    polish_maxiter: int = 500,
    polish_maxfun: int = 650,
    initial_parameters: np.ndarray | None = None,
    initialization: str = "independent",
) -> TransportFit:
    """Two-stage genuine VI: AMSGrad followed by fixed-Sobol L-BFGS."""
    dimension = target.dimension
    rank = family_rank(family)
    random = np.random.default_rng(seed)
    if initial_parameters is None:
        parameters = initial_transport_parameters(
            target, family, seed, map_name, map_scale
        )
    else:
        parameters = np.asarray(
            initial_parameters, dtype=np.float64
        ).copy()
        if len(parameters) != transport_parameter_count(
            dimension, family
        ):
            raise ValueError("warm-start parameter size mismatch")
    stage_scales = transport_optimization_scales(
        target, family, map_name, map_scale
    )
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    maximum_second_moment = np.zeros_like(parameters)
    beta1, beta2 = 0.9, 0.999
    validation_standard = sobol_standard_normal(
        validation_samples,
        dimension + max(rank, 0),
        seed + 10_000_019,
    )
    validation_diagonal, validation_rank = split_noise(
        validation_standard, dimension, rank
    )
    best_parameters = parameters.copy()
    best_validation = -np.inf
    history: list[dict[str, float]] = []
    stable_windows = 0
    previous_validation: float | None = None
    last_substantial_improvement_step = 0
    substantial_best = -np.inf
    failure_reason = ""
    start_time = time.perf_counter()
    adam_steps = 0
    for step in range(1, max_steps + 1):
        diagonal_noise = random.standard_normal(
            (batch_size, dimension)
        )
        rank_noise = (
            random.standard_normal((batch_size, rank))
            if rank > 0
            else None
        )
        train_value, gradient = elbo_and_gradient(
            parameters,
            target,
            family,
            diagonal_noise,
            rank_noise,
            map_name,
            map_scale,
        )
        if (
            not np.isfinite(train_value)
            or not np.all(np.isfinite(gradient))
        ):
            failure_reason = (
                f"non-finite objective/gradient at Adam step {step}"
            )
            break
        scaled_gradient = gradient * stage_scales
        gradient_norm = float(np.linalg.norm(scaled_gradient))
        if gradient_norm > 100.0:
            scaled_gradient *= 100.0 / gradient_norm
        first_moment = (
            beta1 * first_moment
            + (1.0 - beta1) * scaled_gradient
        )
        second_moment = (
            beta2 * second_moment
            + (1.0 - beta2)
            * scaled_gradient
            * scaled_gradient
        )
        maximum_second_moment = np.maximum(
            maximum_second_moment, second_moment
        )
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_maximum = (
            maximum_second_moment / (1.0 - beta2**step)
        )
        current_learning_rate = learning_rate
        parameters += (
            current_learning_rate
            * stage_scales
            * corrected_first
            / (np.sqrt(corrected_maximum) + 1e-8)
        )
        parameters = project_transport_parameters(
            parameters, dimension, family
        )
        adam_steps = step
        if (
            step == 1
            or step % check_every == 0
            or step == max_steps
        ):
            validation, validation_gradient = elbo_and_gradient(
                parameters,
                target,
                family,
                validation_diagonal,
                validation_rank,
                map_name,
                map_scale,
            )
            normalized_gradient = float(
                np.linalg.norm(
                    validation_gradient * stage_scales
                )
                / (
                    1.0
                    + np.linalg.norm(parameters / stage_scales)
                )
            )
            relative_change = (
                float("inf")
                if previous_validation is None
                else abs(validation - previous_validation)
                / (1.0 + abs(previous_validation))
            )
            history.append(
                {
                    "stage": 1.0,
                    "step": float(step),
                    "train_elbo": float(train_value),
                    "validation_elbo": float(validation),
                    "normalized_gradient": normalized_gradient,
                    "relative_change": relative_change,
                }
            )
            if validation > best_validation:
                best_validation = float(validation)
                best_parameters = parameters.copy()
            if validation > substantial_best + 1e-3:
                substantial_best = float(validation)
                last_substantial_improvement_step = step
            if (
                previous_validation is not None
                and relative_change < 1e-4
            ):
                stable_windows += 1
            else:
                stable_windows = 0
            previous_validation = float(validation)
            if stable_windows >= 5:
                break
            # Stage A is an initializer for the fixed-Sobol second stage.  A
            # long stochastic tail after 1,000 steps adds compute but no useful
            # initialization once the best checkpoint has not improved for
            # five check intervals.
            if (
                step >= 800
                and step - last_substantial_improvement_step
                >= 3 * check_every
            ):
                break
    plateau_passed = stable_windows >= 5
    lbfgs_iterations = 0
    lbfgs_function_evaluations = 0
    if not failure_reason and polish_maxiter > 0:
        polish_standard = sobol_standard_normal(
            polish_samples,
            dimension + max(rank, 0),
            seed + 20_000_033,
        )
        polish_diagonal, polish_rank = split_noise(
            polish_standard, dimension, rank
        )

        polish_center = best_parameters.copy()
        polish_scales = transport_optimization_scales(
            target, family, map_name, map_scale
        )
        physical_bounds = transport_parameter_bounds(
            dimension, family
        )
        scaled_bounds = [
            (
                (lower - center) / scale,
                (upper - center) / scale,
            )
            for (lower, upper), center, scale in zip(
                physical_bounds, polish_center, polish_scales
            )
        ]
        latest_objective = float("nan")
        polish_validation_trace: list[float] = []

        def objective(
            scaled_current: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal latest_objective
            current = polish_center + polish_scales * scaled_current
            value, gradient = elbo_and_gradient(
                current,
                target,
                family,
                polish_diagonal,
                polish_rank,
                map_name,
                map_scale,
            )
            latest_objective = float(value)
            return -value, -(gradient * polish_scales)

        def callback(_: np.ndarray) -> None:
            # L-BFGS calls the objective immediately before this callback, so
            # the cached fixed-Sobol value is a genuine optimization window
            # without an extra 8192-sample evaluation.
            if np.isfinite(latest_objective):
                polish_validation_trace.append(latest_objective)

        polished = minimize(
            objective,
            np.zeros_like(best_parameters),
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            bounds=scaled_bounds,
            options={
                "maxiter": polish_maxiter,
                "ftol": 1e-14,
                "gtol": 1e-7,
                "maxls": 50,
                "maxcor": 50,
                # Bound pathological line-search evaluations independently of
                # the preregistered 500 iteration ceiling.  Typical converged
                # fits use fewer evaluations; hitting this guard is retained
                # as an optimizer failure rather than hidden.
                "maxfun": polish_maxfun,
            },
        )
        lbfgs_iterations = int(polished.nit)
        lbfgs_function_evaluations = int(polished.nfev)
        polished_parameters = (
            polish_center + polish_scales * polished.x
        )
        if len(polish_validation_trace) >= 6:
            recent_changes = [
                abs(current - previous) / (1.0 + abs(previous))
                for previous, current in zip(
                    polish_validation_trace[-6:-1],
                    polish_validation_trace[-5:],
                )
            ]
            plateau_passed = bool(
                all(change < 1e-4 for change in recent_changes)
            )
        polished_validation, polished_gradient = elbo_and_gradient(
            polished_parameters,
            target,
            family,
            validation_diagonal,
            validation_rank,
            map_name,
            map_scale,
        )
        polished_normalized_gradient = float(
            np.linalg.norm(
                polished_gradient * stage_scales
            )
            / (
                1.0
                + np.linalg.norm(
                    polished_parameters / stage_scales
                )
            )
        )
        history.append(
            {
                "stage": 2.0,
                "step": float(adam_steps + lbfgs_iterations),
                "train_elbo": float(-polished.fun),
                "validation_elbo": float(polished_validation),
                "normalized_gradient": polished_normalized_gradient,
                "relative_change": float(
                    abs(polished_validation - best_validation)
                    / (1.0 + abs(best_validation))
                ),
            }
        )
        if (
            np.isfinite(polished_validation)
            and polished_validation > best_validation
        ):
            best_parameters = polished_parameters.copy()
            best_validation = float(polished_validation)
    final_validation, final_gradient = elbo_and_gradient(
        best_parameters,
        target,
        family,
        validation_diagonal,
        validation_rank,
        map_name,
        map_scale,
    )
    normalized_gradient = float(
        np.linalg.norm(final_gradient * stage_scales)
        / (
            1.0
            + np.linalg.norm(best_parameters / stage_scales)
        )
    )
    provisional = TransportFit(
        family=family,
        rank=rank,
        map_name=map_name,
        map_scale=float(map_scale),
        parameters=best_parameters,
        validation_elbo=float(final_validation),
        normalized_gradient=normalized_gradient,
        adam_steps=adam_steps,
        lbfgs_iterations=lbfgs_iterations,
        wall_time_seconds=time.perf_counter() - start_time,
        plateau_passed=plateau_passed,
        gradient_passed=normalized_gradient < 1e-2,
        independent_elbo_1=float("nan"),
        independent_elbo_1_se=float("nan"),
        independent_elbo_2=float("nan"),
        independent_elbo_2_se=float("nan"),
        independent_agreement_passed=False,
        converged=False,
        failure_reason=failure_reason,
        history=history,
        initialization=initialization,
        lbfgs_function_evaluations=lbfgs_function_evaluations,
    )
    evaluation_count = max(8192, validation_samples)
    first_elbo, first_se = independent_elbo_estimate(
        provisional,
        target,
        evaluation_count,
        seed + 30_000_059,
    )
    second_elbo, second_se = independent_elbo_estimate(
        provisional,
        target,
        evaluation_count,
        seed + 40_000_063,
    )
    agreement = abs(first_elbo - second_elbo) <= 3.0 * math.sqrt(
        first_se * first_se + second_se * second_se
    )
    provisional.independent_elbo_1 = first_elbo
    provisional.independent_elbo_1_se = first_se
    provisional.independent_elbo_2 = second_elbo
    provisional.independent_elbo_2_se = second_se
    provisional.independent_agreement_passed = bool(agreement)
    provisional.converged = bool(
        not failure_reason
        and provisional.plateau_passed
        and provisional.gradient_passed
        and provisional.independent_agreement_passed
    )
    if not provisional.converged and not provisional.failure_reason:
        failed = []
        if not provisional.plateau_passed:
            failed.append("validation_plateau")
        if not provisional.gradient_passed:
            failed.append("normalized_gradient")
        if not provisional.independent_agreement_passed:
            failed.append("independent_elbo_agreement")
        provisional.failure_reason = ",".join(failed)
    provisional.wall_time_seconds = time.perf_counter() - start_time
    return provisional


def sample_transport(
    fit: TransportFit,
    count: int,
    seed: int,
    chunk_size: int | None = None,
) -> np.ndarray:
    dimension = (
        len(fit.parameters) // (2 + max(fit.rank, 0))
        if fit.rank >= 0
        else int(
            round(
                (-3.0 + math.sqrt(9.0 + 8.0 * len(fit.parameters)))
                / 2.0
            )
        )
    )
    if fit.family == "full":
        while (
            dimension + dimension * (dimension + 1) // 2
            != len(fit.parameters)
        ):
            dimension += 1
    chunk_size = count if chunk_size is None else chunk_size
    random = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, count, chunk_size):
        current = min(chunk_size, count - start)
        diagonal_noise = random.standard_normal(
            (current, dimension)
        )
        rank_noise = (
            random.standard_normal((current, fit.rank))
            if fit.rank > 0
            else None
        )
        auxiliary = draw_auxiliary(
            fit.parameters,
            dimension,
            fit.family,
            diagonal_noise,
            rank_noise,
        )
        chunks.append(
            positive_map_forward(
                auxiliary, fit.map_name, fit.map_scale
            )
        )
    return np.vstack(chunks)


def expand_warm_start(
    source: TransportFit,
    target_family: str,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Embed diag/rank-r parameters into a larger nested low-rank family."""
    target_rank = family_rank(target_family)
    if source.rank < 0 or target_rank < source.rank:
        raise ValueError("warm start requires nested low-rank families")
    mean, log_diagonal, low_rank = unpack_low_rank(
        source.parameters, dimension, source.rank
    )
    random = np.random.default_rng(seed)
    expanded = np.zeros((dimension, target_rank), dtype=np.float64)
    if source.rank:
        expanded[:, : source.rank] = low_rank
    if target_rank > source.rank:
        expanded[:, source.rank :] = (
            1e-5
            * random.standard_normal(
                (dimension, target_rank - source.rank)
            )
        )
    return np.concatenate(
        [mean, log_diagonal, expanded.ravel()]
    )


def warm_start_full(
    source: TransportFit, dimension: int
) -> np.ndarray:
    """Embed a selected low-rank Gaussian exactly into the full family."""
    if source.rank < 0:
        return source.parameters.copy()
    mean, _, _ = unpack_low_rank(
        source.parameters, dimension, source.rank
    )
    covariance = transport_covariance(
        source.parameters, dimension, source.family
    )
    lower, _ = stable_cholesky(covariance, jitter=1e-12)
    rows, columns = lower_indices(dimension)
    raw = lower[rows, columns].copy()
    diagonal_positions = rows == columns
    raw[diagonal_positions] = np.log(
        raw[diagonal_positions]
    )
    return np.concatenate([mean, raw])


def ghk_log_orthant(
    target: GaussianTarget,
    seeds: list[int],
    samples_per_seed: int,
) -> dict[str, Any]:
    """Randomized-Sobol GHK estimate of log P(D >= 0)."""
    dimension = target.dimension
    power = int(math.ceil(math.log2(samples_per_seed)))
    count = 2**power
    estimates: list[float] = []
    for seed in seeds:
        engine = qmc.Sobol(
            d=dimension, scramble=True, seed=seed
        )
        uniforms = engine.random_base2(power)
        latent = np.zeros(
            (count, dimension), dtype=np.float64
        )
        log_weights = np.zeros(count, dtype=np.float64)
        for coordinate in range(dimension):
            conditional_mean = np.full(
                count,
                target.mean[coordinate],
                dtype=np.float64,
            )
            if coordinate:
                conditional_mean += (
                    latent[:, :coordinate]
                    @ target.cholesky[coordinate, :coordinate]
                )
            standardized_lower = (
                -conditional_mean
                / target.cholesky[coordinate, coordinate]
            )
            log_probability = log_ndtr(
                -standardized_lower
            )
            probability = np.exp(log_probability)
            cdf = (
                1.0
                - (1.0 - uniforms[:, coordinate]) * probability
            )
            cdf = np.clip(
                cdf,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            latent[:, coordinate] = ndtri(cdf)
            log_weights += log_probability
        estimates.append(
            float(logsumexp(log_weights) - math.log(count))
        )
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "method": "randomized_sobol_ghk",
        "seeds": [int(seed) for seed in seeds],
        "samples_per_seed": count,
        "logz_estimates": estimates,
        "logz": float(np.mean(values)),
        "logz_se": (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        ),
    }


def estimate_log_orthant(
    target: GaussianTarget,
    samples_per_seed: int = 32768,
    replicates: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate and independently cross-check the hard normalizer."""
    ghk = ghk_log_orthant(
        target,
        [seed + 104729 * index for index in range(replicates)],
        samples_per_seed,
    )
    scipy_estimates: list[float] = []
    for index in range(3):
        # SciPy 1.12 uses NumPy's legacy global stream internally and does not
        # yet expose an ``rng`` argument for multivariate_normal.cdf.
        np.random.seed(seed + 99_991 + index)
        probability = multivariate_normal.cdf(
            np.zeros(target.dimension),
            mean=-target.mean,
            cov=target.covariance,
            maxpts=500_000 * target.dimension,
            abseps=1e-8,
            releps=1e-8,
        )
        scipy_estimates.append(
            float(math.log(max(float(probability), TINY)))
        )
    scipy_values = np.asarray(
        scipy_estimates, dtype=np.float64
    )
    scipy_mean = float(np.mean(scipy_values))
    scipy_se = (
        float(
            np.std(scipy_values, ddof=1)
            / math.sqrt(len(scipy_values))
        )
        if len(scipy_values) > 1
        else 0.0
    )
    discrepancy = abs(float(ghk["logz"]) - scipy_mean)
    tolerance = max(
        0.05,
        5.0
        * math.sqrt(
            float(ghk["logz_se"]) ** 2 + scipy_se**2
        ),
    )
    return {
        "primary": ghk,
        "crosscheck": {
            "method": "scipy_mvn_cdf",
            "logz_estimates": scipy_estimates,
            "logz": scipy_mean,
            "logz_se": scipy_se,
        },
        "absolute_logz_discrepancy": discrepancy,
        "reliability_tolerance": tolerance,
        "reliable": bool(
            np.isfinite(ghk["logz"])
            and discrepancy <= tolerance
        ),
    }


def rejection_reference(
    target: GaussianTarget,
    count: int,
    seed: int,
    max_draws: int = 20_000_000,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    random = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    retained = 0
    draws = 0
    while retained < count and draws < max_draws:
        current = min(batch_size, max_draws - draws)
        values = (
            target.mean[None, :]
            + random.standard_normal(
                (current, target.dimension)
            )
            @ target.cholesky.T
        )
        keep = values[np.all(values >= 0.0, axis=1)]
        if len(keep):
            accepted.append(keep)
            retained += len(keep)
        draws += current
    if retained < count:
        raise RuntimeError(
            f"rejection retained {retained}/{count} after {draws}"
        )
    output = np.vstack(accepted)[:count]
    return output, {
        "method": "rejection",
        "draws": draws,
        "retained": count,
        "observed_acceptance": retained / draws,
        "split_rhat_max": 1.0,
        "ess_min": float(count),
        "reliable": True,
    }


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Classical split-Rhat for chains x draws x dimensions."""
    chain_count, draw_count, _ = chains.shape
    half = draw_count // 2
    if chain_count < 2 or half < 4:
        return np.full(chains.shape[2], np.inf)
    split = np.concatenate(
        [chains[:, :half], chains[:, -half:]], axis=0
    )
    within_variance = np.mean(
        np.var(split, axis=1, ddof=1), axis=0
    )
    between_variance = half * np.var(
        np.mean(split, axis=1), axis=0, ddof=1
    )
    variance = (
        (half - 1.0) / half * within_variance
        + between_variance / half
    )
    return np.sqrt(
        np.maximum(variance / np.maximum(within_variance, TINY), 0.0)
    )


def batch_means_ess(chains: np.ndarray) -> np.ndarray:
    """Conservative batch-means ESS diagnostic."""
    chain_count, draw_count, dimension = chains.shape
    flattened = chains.reshape(chain_count * draw_count, dimension)
    total = len(flattened)
    batch_size = max(10, int(math.sqrt(total)))
    batch_count = total // batch_size
    trimmed = flattened[: batch_count * batch_size]
    marginal_variance = np.var(trimmed, axis=0, ddof=1)
    batch_means = trimmed.reshape(
        batch_count, batch_size, dimension
    ).mean(axis=1)
    asymptotic_variance = batch_size * np.var(
        batch_means, axis=0, ddof=1
    )
    return np.clip(
        total
        * marginal_variance
        / np.maximum(asymptotic_variance, TINY),
        1.0,
        float(total),
    )


def gibbs_reference(
    target: GaussianTarget,
    count: int,
    seed: int,
    chains: int = 8,
    burn_in: int = 2500,
    thin: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinate Gibbs sampler for a nonnegative truncated Gaussian."""
    dimension = target.dimension
    retained_per_chain = int(math.ceil(count / chains))
    total_sweeps = burn_in + retained_per_chain * thin
    random = np.random.default_rng(seed)
    precision = cho_solve(
        target.factor, np.eye(dimension), check_finite=False
    )
    conditional_sd = np.sqrt(
        1.0 / np.diag(precision)
    )
    marginal_sd = np.sqrt(
        np.diag(target.covariance)
    )
    state = np.maximum(
        target.mean[None, :]
        + 0.25
        * random.standard_normal((chains, dimension))
        * marginal_sd[None, :],
        0.05 * marginal_sd[None, :],
    )
    retained = np.empty(
        (chains, retained_per_chain, dimension),
        dtype=np.float64,
    )
    write_index = 0
    for sweep in range(total_sweeps):
        difference = state - target.mean[None, :]
        for coordinate in range(dimension):
            off_diagonal_effect = (
                difference @ precision[:, coordinate]
                - difference[:, coordinate]
                * precision[coordinate, coordinate]
            )
            conditional_mean = (
                target.mean[coordinate]
                - off_diagonal_effect
                / precision[coordinate, coordinate]
            )
            standardized_lower = (
                -conditional_mean / conditional_sd[coordinate]
            )
            lower_cdf = ndtr(standardized_lower)
            uniform = lower_cdf + (
                1.0 - lower_cdf
            ) * random.random(chains)
            uniform = np.clip(
                uniform,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            state[:, coordinate] = (
                conditional_mean
                + conditional_sd[coordinate] * ndtri(uniform)
            )
            difference[:, coordinate] = (
                state[:, coordinate] - target.mean[coordinate]
            )
        if (
            sweep >= burn_in
            and (sweep - burn_in) % thin == 0
        ):
            retained[:, write_index, :] = state
            write_index += 1
    rhat = split_rhat(retained)
    ess = batch_means_ess(retained)
    output = retained.reshape(-1, dimension)[:count]
    reliable = bool(
        np.max(rhat) < 1.05
        and np.min(ess) >= min(400.0, 0.05 * count)
        and np.all(output >= 0.0)
    )
    return output, {
        "method": "coordinate_gibbs",
        "chains": chains,
        "burn_in": burn_in,
        "thin": thin,
        "retained": count,
        "split_rhat_max": float(np.max(rhat)),
        "ess_min": float(np.min(ess)),
        "reliable": reliable,
    }


def hard_reference_samples(
    target: GaussianTarget,
    count: int,
    seed: int,
    logz: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use exact rejection when practical and Gibbs otherwise."""
    expected_acceptance = (
        math.exp(logz) if logz is not None and logz > -50.0 else 0.0
    )
    expected_draws = (
        count / expected_acceptance
        if expected_acceptance > 0.0
        else float("inf")
    )
    # Rejection cost is controlled by the orthant probability, not directly
    # by dimension.  The previous dimension <= 8 guard unnecessarily routed
    # high-probability m=10/20 targets to a slowly mixing Gibbs sampler.
    # The explicit expected-draw budget is the relevant safety condition.
    if expected_draws <= 5_000_000:
        return rejection_reference(target, count, seed)
    return gibbs_reference(target, count, seed)


def hard_logpdf(
    target: GaussianTarget, logz: float, values: np.ndarray
) -> np.ndarray:
    values = np.atleast_2d(
        np.asarray(values, dtype=np.float64)
    )
    inside = np.all(values >= 0.0, axis=1)
    output = np.full(len(values), -np.inf, dtype=np.float64)
    output[inside] = target.logpdf(values[inside]) - logz
    return output


def gaussian_samples(
    target: GaussianTarget, count: int, seed: int
) -> np.ndarray:
    random = np.random.default_rng(seed)
    return (
        target.mean[None, :]
        + random.standard_normal((count, target.dimension))
        @ target.cholesky.T
    )


def forward_kl_from_reference(
    target: GaussianTarget,
    logz: float,
    reference: np.ndarray,
    candidate_logpdf: np.ndarray,
) -> tuple[float, float]:
    terms = (
        target.logpdf(reference)
        - logz
        - np.asarray(candidate_logpdf, dtype=np.float64)
    )
    if not np.all(np.isfinite(terms)):
        return float("inf"), 0.0
    return float(np.mean(terms)), float(
        np.std(terms, ddof=1) / math.sqrt(len(terms))
    )


def mixture_tv(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    p_logpdf,
    q_logpdf,
) -> tuple[float, float]:
    """Equal-mixture importance estimator of total variation."""
    values = np.vstack([p_samples, q_samples])
    log_p = np.asarray(p_logpdf(values), dtype=np.float64)
    log_q = np.asarray(q_logpdf(values), dtype=np.float64)
    denominator = np.logaddexp(log_p, log_q)
    p_ratio = np.exp(log_p - denominator)
    q_ratio = np.exp(log_q - denominator)
    contributions = np.abs(p_ratio - q_ratio)
    estimate = float(np.mean(contributions))
    standard_error = float(
        np.std(contributions, ddof=1)
        / math.sqrt(len(contributions))
    )
    return estimate, standard_error


def average_marginal_w1(
    reference: np.ndarray, candidate: np.ndarray
) -> float:
    return float(
        np.mean(
            [
                wasserstein_distance(
                    reference[:, coordinate],
                    candidate[:, coordinate],
                )
                for coordinate in range(reference.shape[1])
            ]
        )
    )


def sliced_w1(
    reference: np.ndarray,
    candidate: np.ndarray,
    projections: int,
    seed: int,
    max_samples: int = 10_000,
) -> float:
    random = np.random.default_rng(seed)
    count = min(len(reference), len(candidate), max_samples)
    reference_indices = random.choice(
        len(reference), count, replace=False
    )
    candidate_indices = random.choice(
        len(candidate), count, replace=False
    )
    directions = random.standard_normal(
        (reference.shape[1], projections)
    )
    directions /= np.linalg.norm(
        directions, axis=0, keepdims=True
    )
    projected_reference = np.sort(
        reference[reference_indices] @ directions, axis=0
    )
    projected_candidate = np.sort(
        candidate[candidate_indices] @ directions, axis=0
    )
    return float(
        np.mean(
            np.abs(projected_reference - projected_candidate)
        )
    )


def wasserstein_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    projections: int,
    seed: int,
) -> dict[str, float]:
    return {
        "average_marginal_w1": average_marginal_w1(
            reference, candidate
        ),
        "sliced_w1": sliced_w1(
            reference, candidate, projections, seed
        ),
    }


def sample_moment_errors(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    reference_mean = np.mean(reference, axis=0)
    candidate_mean = np.mean(candidate, axis=0)
    reference_covariance = np.cov(reference, rowvar=False)
    candidate_covariance = np.cov(candidate, rowvar=False)
    return {
        "mean_relative_error": float(
            np.linalg.norm(candidate_mean - reference_mean)
            / max(np.linalg.norm(reference_mean), 1e-12)
        ),
        "covariance_relative_error": float(
            np.linalg.norm(
                candidate_covariance - reference_covariance,
                ord="fro",
            )
            / max(
                np.linalg.norm(
                    reference_covariance, ord="fro"
                ),
                1e-12,
            )
        ),
    }


def evaluate_gaussian_method(
    method: str,
    candidate: GaussianTarget,
    hard_target: GaussianTarget,
    logz: float,
    reference: np.ndarray,
    candidate_count: int,
    tv_count: int,
    projections: int,
    seed: int,
) -> dict[str, Any]:
    candidate_samples = gaussian_samples(
        candidate, candidate_count, seed
    )
    forward, forward_se = forward_kl_from_reference(
        hard_target,
        logz,
        reference,
        candidate.logpdf(reference),
    )
    if method in {"Standard GP", "Projection GP"}:
        tv = 1.0 - math.exp(logz)
        tv_se = 0.0
    else:
        p_subset = reference[: min(tv_count, len(reference))]
        q_subset = gaussian_samples(
            candidate, len(p_subset), seed + 1
        )
        tv, tv_se = mixture_tv(
            p_subset,
            q_subset,
            lambda values: hard_logpdf(
                hard_target, logz, values
            ),
            candidate.logpdf,
        )
    result: dict[str, Any] = {
        "method": method,
        "reverse_kl": float("inf"),
        "reverse_kl_se": 0.0,
        "forward_kl": forward,
        "forward_kl_se": forward_se,
        "tv": tv,
        "tv_se": tv_se,
    }
    result.update(
        wasserstein_metrics(
            reference,
            candidate_samples,
            projections,
            seed + 2,
        )
    )
    result.update(
        sample_moment_errors(reference, candidate_samples)
    )
    return result


def evaluate_basis_method(
    point: np.ndarray,
    reference: np.ndarray,
    projections: int,
    seed: int,
) -> dict[str, Any]:
    candidate = np.repeat(
        np.asarray(point, dtype=np.float64)[None, :],
        len(reference),
        axis=0,
    )
    result: dict[str, Any] = {
        "method": "Basis GP",
        "reverse_kl": float("inf"),
        "reverse_kl_se": 0.0,
        "forward_kl": float("inf"),
        "forward_kl_se": 0.0,
        "tv": 1.0,
        "tv_se": 0.0,
    }
    result.update(
        wasserstein_metrics(
            reference, candidate, projections, seed
        )
    )
    result.update(sample_moment_errors(reference, candidate))
    return result


def evaluate_transport_method(
    fit: TransportFit,
    hard_target: GaussianTarget,
    logz: float,
    reference: np.ndarray,
    elbo_samples: int,
    candidate_count: int,
    tv_count: int,
    projections: int,
    seed: int,
) -> dict[str, Any]:
    elbo, elbo_se = independent_elbo_estimate(
        fit, hard_target, elbo_samples, seed
    )
    candidate = sample_transport(
        fit, candidate_count, seed + 1
    )
    forward, forward_se = forward_kl_from_reference(
        hard_target,
        logz,
        reference,
        transport_logpdf_derivative(fit, reference),
    )
    p_subset = reference[: min(tv_count, len(reference))]
    q_subset = sample_transport(
        fit, len(p_subset), seed + 2
    )
    tv, tv_se = mixture_tv(
        p_subset,
        q_subset,
        lambda values: hard_logpdf(
            hard_target, logz, values
        ),
        lambda values: transport_logpdf_derivative(
            fit, values
        ),
    )
    auxiliary_tail_sample_count = min(100_000, candidate_count)
    random = np.random.default_rng(seed + 3)
    dimension = hard_target.dimension
    diagonal_noise = random.standard_normal(
        (auxiliary_tail_sample_count, dimension)
    )
    rank_noise = (
        random.standard_normal(
            (auxiliary_tail_sample_count, fit.rank)
        )
        if fit.rank > 0
        else None
    )
    auxiliary = draw_auxiliary(
        fit.parameters,
        dimension,
        fit.family,
        diagonal_noise,
        rank_noise,
    )
    log_jacobian = positive_map_log_jacobian(
        auxiliary, fit.map_name, fit.map_scale
    )
    result: dict[str, Any] = {
        "method": "CTVGP",
        "reverse_kl": float(logz - elbo),
        "reverse_kl_se": elbo_se,
        "forward_kl": forward,
        "forward_kl_se": forward_se,
        "tv": tv,
        "tv_se": tv_se,
        "validation_elbo": fit.validation_elbo,
        "normalized_gradient": fit.normalized_gradient,
        "converged": fit.converged,
        "wall_time_seconds": fit.wall_time_seconds,
        "jacobian_log_q01": float(
            np.quantile(log_jacobian, 0.01)
        ),
        "jacobian_log_q50": float(
            np.quantile(log_jacobian, 0.50)
        ),
        "jacobian_log_q99": float(
            np.quantile(log_jacobian, 0.99)
        ),
        "minimum_derivative": float(np.min(candidate)),
    }
    result.update(
        wasserstein_metrics(
            reference, candidate, projections, seed + 4
        )
    )
    result.update(sample_moment_errors(reference, candidate))
    return result


def bootstrap_mean_interval(
    values: np.ndarray,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        return {
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": int(len(values)),
        }
    random = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, replicates, chunk_size):
        current = min(chunk_size, replicates - start)
        indices = random.integers(
            0, len(values), size=(current, len(values))
        )
        bootstrap[start : start + current] = np.mean(
            values[indices], axis=1
        )
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "n": int(len(values)),
    }
