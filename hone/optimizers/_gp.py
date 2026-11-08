"""Gaussian process regression, the model behind Bayesian optimization.

The kernel is Matérn 5/2 with one length scale per input, so the model can learn that some
inputs matter more than others. Inputs are expected in [0, 1]; outputs are standardized.
Hyperparameters are chosen by maximizing the marginal likelihood times weak priors (MAP).
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lapack, solve_triangular
from scipy.optimize import minimize

SQRT5 = math.sqrt(5)

# Bounds on the log hyperparameters: length scales, signal variance, noise variance.
LOG_LENGTH = (math.log(0.01), math.log(20.0))
LOG_SIGNAL = (math.log(0.05), math.log(20.0))
LOG_NOISE = (math.log(1e-6), math.log(1.0))

# Weak priors on the log hyperparameters, as (mean, standard deviation) of a normal.
LENGTH_PRIOR = (math.log(0.3), 1.0)  # inputs are in [0, 1], so most length scales are a fraction of that
SIGNAL_PRIOR = (0.0, 1.0)  # outputs are standardized
NOISE_PRIOR = (math.log(1e-4), 2.0)  # little noise unless the data says otherwise


# Where an array starts in memory can change the order in which a BLAS library adds up its
# elements. Apple's Accelerate does this: a dot product, a product of a matrix and a vector, or a
# Cholesky solve for one vector can differ in the last bit between an operand on a 256-byte
# boundary and one that isn't, from about 64 rows up. NumPy arrays under about 1,000 bytes get
# whatever address the allocator has free, which differs from call to call, and the optimizers
# amplify a last-bit difference into different candidates, so a seeded run stopped repeating.
# The vectors those operations see are therefore placed on a 256-byte boundary; larger arrays
# already are, on macOS, and are copied only if not.
ALIGNMENT = 256


def _aligned(a: np.ndarray, copy: bool = False) -> np.ndarray:
    """a itself if its data is contiguous and starts on an ALIGNMENT-byte boundary, else, or with copy, a copy that does."""
    a = np.asarray(a, dtype=float)
    if not copy and a.ctypes.data % ALIGNMENT == 0 and (a.flags.c_contiguous or a.flags.f_contiguous):
        return a
    buffer = np.empty(a.size + ALIGNMENT // 8)
    start = (-buffer.ctypes.data % ALIGNMENT) // 8
    out = buffer[start : start + a.size].reshape(a.shape)
    out[...] = a
    return out


def _solve_vector(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """K^-1 b for K = L L^T, L lower triangular, with both operands on an ALIGNMENT-byte boundary."""
    if L.flags.f_contiguous:
        factor, lower = _aligned(L), 1
    else:  # a C-ordered lower triangle is a Fortran-ordered upper one, with K = U^T U for U = L^T
        factor, lower = _aligned(L).T, 0
    x, info = lapack.dpotrs(factor, _aligned(b, copy=True), lower=lower, overwrite_b=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotrs failed with info {info}")
    return _aligned(x)


def _matern(r: np.ndarray) -> np.ndarray:
    return (1 + SQRT5 * r + 5 / 3 * r**2) * np.exp(-SQRT5 * r)


def _scaled_distances(a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
    a, b = a / length, b / length
    squared = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(squared, 0))


class GaussianProcess:
    """A Gaussian process fitted to inputs X, shape (n, d) in [0, 1], and outputs y, shape (n,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray | None = None) -> None:
        self.X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._offset = y.mean()
        self._scale = y.std() if y.std() > 0 else 1.0
        self.z = _aligned((y - self._offset) / self._scale)
        d = self.X.shape[1]
        self.theta = theta if theta is not None else np.concatenate([np.full(d, LENGTH_PRIOR[0]), [0.0, math.log(1e-4)]])
        self._condition()

    # --- hyperparameters ------------------------------------------------------

    def fit(self, restarts: int = 0, rng: np.random.Generator | None = None) -> "GaussianProcess":
        """Choose hyperparameters by MAP, starting from the current ones and restarts random others."""
        d = self.X.shape[1]
        bounds = [LOG_LENGTH] * d + [LOG_SIGNAL, LOG_NOISE]
        starts = [self.theta]
        for _ in range(restarts):
            assert rng is not None, "restarts need a random number generator"
            starts.append(np.array([rng.uniform(low, high) for low, high in bounds]))
        fits = [
            minimize(self._objective, np.clip(start, *np.array(bounds).T), jac=True, method="L-BFGS-B", bounds=bounds)
            for start in starts
        ]
        self.theta = min(fits, key=lambda found: found.fun).x
        self._condition()
        return self

    def _objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """The negative log posterior of theta, and its gradient."""
        d = self.X.shape[1]
        length, signal, noise = np.exp(theta[:d]), math.exp(theta[d]), math.exp(theta[d + 1])
        r = _scaled_distances(self.X, self.X, length)
        correlation = _matern(r)
        K = signal * correlation + (noise + 1e-10) * np.eye(len(self.X))
        try:
            factor = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        alpha = _solve_vector(factor[0], self.z)
        L = factor[0]
        nll = 0.5 * self.z @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(self.z) * math.log(2 * math.pi)

        W = np.outer(alpha, alpha) - cho_solve(factor, np.eye(len(self.X)))
        # d k / d log(length_j) = signal * 5/3 * (1 + sqrt5 r) exp(-sqrt5 r) * (dx_j / length_j)^2
        S = W * (signal * 5 / 3 * (1 + SQRT5 * r) * np.exp(-SQRT5 * r))
        scaled = self.X / length
        length_gradient = 2 * (S.sum(1) @ scaled**2) - 2 * np.einsum("ij,ik,kj->j", scaled, S, scaled)
        gradient = -0.5 * np.concatenate(
            [length_gradient, [np.sum(W * signal * correlation), noise * np.trace(W)]]
        )

        for index, (mean, sd) in [(slice(0, d), LENGTH_PRIOR), (d, SIGNAL_PRIOR), (d + 1, NOISE_PRIOR)]:
            nll += 0.5 * np.sum(((theta[index] - mean) / sd) ** 2)
            gradient[index] += (theta[index] - mean) / sd**2
        return float(nll), gradient

    # --- prediction -----------------------------------------------------------

    def _condition(self) -> None:
        d = self.X.shape[1]
        self._length, self._signal = np.exp(self.theta[:d]), math.exp(self.theta[d])
        noise = math.exp(self.theta[d + 1])
        K = self._signal * _matern(_scaled_distances(self.X, self.X, self._length))
        K[np.diag_indices_from(K)] += noise + 1e-10
        jitter = 0.0
        while True:
            try:
                self._L = np.linalg.cholesky(K + jitter * np.eye(len(K)))
                break
            except np.linalg.LinAlgError:
                jitter = max(1e-8, jitter * 10)
        self._alpha = _solve_vector(self._L, self.z)
        # Predictions multiply by the inverse of L, which is faster than solving with L each time.
        self._L_inverse = solve_triangular(self._L, np.eye(len(K)), lower=True)

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at Xs, in standardized units."""
        k = _aligned(self._signal * _matern(_scaled_distances(np.atleast_2d(Xs), self.X, self._length)))
        mean = k @ self._alpha
        v = self._L_inverse @ k.T
        variance = np.maximum(self._signal - (v**2).sum(0), 1e-12)
        return mean, np.sqrt(variance)

    def standardize(self, y: float | np.ndarray) -> float | np.ndarray:
        return (np.asarray(y) - self._offset) / self._scale

    def predict_original(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at Xs, in the units of the observations."""
        mean, sd = self.predict(Xs)
        return mean * self._scale + self._offset, sd * self._scale

    def with_observation(self, x: np.ndarray, z: float) -> "GaussianProcess":
        """This model with one more observation, given in standardized units, and the same hyperparameters."""
        model = object.__new__(GaussianProcess)
        model.X = np.vstack([self.X, x])
        model._offset, model._scale = self._offset, self._scale
        model.z = _aligned(np.append(self.z, z))
        model.theta = self.theta
        model._condition()
        return model
