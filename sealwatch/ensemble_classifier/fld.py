"""Fisher linear discriminant (FLD)."""

from __future__ import annotations

import typing

import numpy as np

from ._implementation import Implementation


class FisherLinearDiscriminant(object):
    """Fisher linear discriminant (FLD) for two classes, covers (0) and stegos (1).

    The weights ``w`` solve a linear system with the within-class scatter matrix.
    The threshold ``b`` minimizes the total error under equal priors on the training set,
    :math:`P_E = \\min_{P_{FA}} \\frac{1}{2} (P_{FA} + P_{MD}(P_{FA}))`.
    The sign of ``w`` is chosen such that stegos score higher than covers.
    Features that are constant in both classes get zero weight.

    A single FLD is weak and unstable, but cheap to train; an ensemble of them
    increases diversity.

    :param implementation: code path, see :class:`~sealwatch.ensemble_classifier.Implementation`.
    :param w: weights of shape ``[num_features]``; ``None`` until fitted.
    :param b: bias (threshold); ``None`` until fitted.
    """

    def __init__(
        self,
        implementation: Implementation = Implementation.FLD_LEGACY,
        w: typing.Optional[np.ndarray] = None,
        b: typing.Optional[float] = None,
    ):
        self.implementation = implementation
        self.w = w
        self.b = b

    def fit(self, X: np.ndarray, y: np.ndarray) -> FisherLinearDiscriminant:
        """Fit on a stacked design matrix.

        :param X: samples of shape ``[num_samples, num_features]``.
        :param y: labels of shape ``[num_samples]``, 0 for covers and 1 for stegos.
        :return: the fitted learner.
        :raises ValueError: if ``y`` does not contain exactly the labels 0 and 1.
        """
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Expected samples with 0 and 1 labels")
        return self.fit_presplit(X[y == 0], X[y == 1])

    def fit_presplit(self, Xc: np.ndarray, Xs: np.ndarray) -> FisherLinearDiscriminant:
        """Fit on covers and stegos that are already separate, same result as :meth:`fit`.

        Saves concatenating them and indexing them apart again.

        :param Xc: cover samples of shape ``[num_covers, num_features]``.
        :param Xs: stego samples of shape ``[num_stegos, num_features]``.
        :return: the fitted learner.
        :raises ValueError: if the implementation is unknown.
        """
        if not isinstance(self.implementation, Implementation):
            raise ValueError(f"Unknown implementation: {self.implementation!r}")
        num_feature_dims = Xc.shape[1]

        # Features that are constant in both classes get zero weight
        drop_feature_dims = self._constant_dims(Xc, Xs)

        # Within-class scatter matrix, made positive definite
        mu, sigma_cs = self._scatter(Xc, Xs)
        sigma_cs = self._regularize(sigma_cs)

        # Check for NaN values (may occur when the feature value is constant over images)
        drop_feature_dims = drop_feature_dims | np.any(np.isnan(sigma_cs), axis=0)

        # Drop feature dimensions from mean and covariance
        sigma_cs = sigma_cs[~drop_feature_dims, :][:, ~drop_feature_dims]
        mu = mu[~drop_feature_dims]

        w = self._solve(sigma_cs, mu)

        if len(w) != num_feature_dims:
            # Resolve previously found NaN columns: Set the corresponding elements of w equal to zero
            w_new = np.zeros(num_feature_dims)
            w_new[~drop_feature_dims] = w
            w = w_new

        # Adjust threshold to minimize the total error under equal priors
        self.w, self.b = self._find_threshold(Xc, Xs, w)
        return self

    @staticmethod
    def _constant_dims(Xc: np.ndarray, Xs: np.ndarray) -> np.ndarray:
        """Find the feature dimensions that hold the same single value in both classes.

        :param Xc: cover samples of shape ``[num_covers, num_features]``.
        :param Xs: stego samples of shape ``[num_stegos, num_features]``.
        :return: boolean mask of shape ``[num_features]``, ``True`` for dimensions to drop.
        """
        drop_feature_dims = np.zeros(Xc.shape[1], dtype=bool)

        drop_dim_candidates = np.unique(np.concatenate([
            np.where(np.all(Xc == Xc[0][None, :], axis=0))[0],
            np.where(np.all(Xs == Xs[0][None, :], axis=0))[0],
        ]))

        for drop_dim_candidate in drop_dim_candidates:
            # Verify number of values in this column
            num_cover_vals = np.unique(Xc[:, drop_dim_candidate])
            if len(num_cover_vals) == 1:
                # Verify that stego images also contain only a single value
                num_stego_vals = np.unique(Xs[:, drop_dim_candidate])
                if len(num_stego_vals) == 1 and num_cover_vals[0] == num_stego_vals[0]:
                    # Flag dimension for dropping
                    drop_feature_dims[drop_dim_candidate] = True

        return drop_feature_dims

    @staticmethod
    def _scatter(Xc: np.ndarray, Xs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Calculate the difference of the class means and the within-class scatter matrix.

        :param Xc: cover samples of shape ``[num_covers, num_features]``.
        :param Xs: stego samples of shape ``[num_stegos, num_features]``.
        :return: ``(mu, sigma_cs)``, the mean difference of shape ``[num_features]`` and
            the scatter matrix of shape ``[num_features, num_features]``, a fresh array.
        """
        # Calculate means of each class
        mu_c = np.mean(Xc, axis=0)
        mu_s = np.mean(Xs, axis=0)
        mu = (mu_s - mu_c).T

        # Calculate covariance for covers
        Xc_zero_mean = Xc - mu_c[None, :]
        sigma_c = Xc_zero_mean.T @ Xc_zero_mean
        sigma_c /= len(Xc)

        # Calculate covariance for stegos
        Xs_zero_mean = Xs - mu_s[None, :]
        sigma_s = Xs_zero_mean.T @ Xs_zero_mean
        sigma_s /= len(Xs)

        # Within-class scatter matrix
        return mu, sigma_c + sigma_s

    @staticmethod
    def _regularize(sigma_cs: np.ndarray) -> np.ndarray:
        """Add a small ridge to ensure that the scatter matrix is positive definite.

        Added in place on the diagonal, which gives the same values as
        ``sigma_cs + ridge * np.eye(d)`` without allocating a d x d identity or
        upcasting float32 to float64.

        :param sigma_cs: scatter matrix of shape ``[num_features, num_features]``, modified in place.
        :return: the regularized scatter matrix.
        """
        num_feature_dims = len(sigma_cs)

        if sigma_cs.dtype == np.float64:
            # Same constant as the legacy code
            ridge = 1e-10
        else:
            # In float32, 1e-10 is far below the machine epsilon (about 1.2e-7) and does not
            # regularize at all; the solve then generates denormals and
            # is about 10x slower. Use a scale-aware ridge instead (about 10x the epsilon,
            # relative to the mean diagonal).
            ridge = 1e-6 * (np.trace(sigma_cs) / num_feature_dims)
        sigma_cs.flat[::num_feature_dims + 1] += ridge
        return sigma_cs

    def _solve(self, sigma_cs: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Solve ``sigma_cs @ w = mu`` for the weights.

        If the matrix is numerically singular, the regularization is increased
        until the solve succeeds.

        :param sigma_cs: symmetric positive definite scatter matrix, modified in place
            if the regularization is increased.
        :param mu: mean difference of shape ``[len(sigma_cs)]``.
        :return: weights of shape ``[len(sigma_cs)]``.
        """
        solve_counter = 0
        while True:
            try:
                # LU, as the legacy code, which keeps the results bit for bit identical.
                # sigma_cs is symmetric positive definite, so Cholesky would also apply
                # (scipy.linalg.solve(sigma_cs, mu, assume_a="pos"); Matlab's mldivide uses it too).
                # We tried it: it changes w in the last bits (more on near-singular matrices)
                # and was not faster for the d_sub of up to about 2000 typical with SRM;
                # it pays off (1.2-1.4x per learner) only from d_sub of about 2500.
                return np.linalg.solve(sigma_cs, mu)
            except np.linalg.LinAlgError:
                # Numerically not positive definite or singular: increase the regularization
                if 0 == solve_counter:
                    solve_counter = 1
                else:
                    solve_counter *= 5

                # Distance from 1 to the next larger representable real number in the working
                # precision. For float64, this is np.spacing(1), as in the legacy code. In float32,
                # the double precision value would vanish next to values of order 1.
                eps = np.finfo(sigma_cs.dtype).eps

                # Dynamically increase stabilizing constant
                sigma_cs += solve_counter * eps * np.eye(len(sigma_cs))

    def _find_threshold(
        self,
        Xc: np.ndarray,
        Xs: np.ndarray,
        w: np.ndarray,
    ) -> typing.Tuple[np.ndarray, float]:
        """Find the threshold that minimizes :math:`(P_{MD} + P_{FA}) / 2`.

        Also decides whether the sign of ``w`` must be flipped, so that stegos score
        higher. Ties go to the first optimum, scanning from low to high scores.

        :param Xc: cover samples of shape ``[num_covers, num_subspace_dims]``.
        :param Xs: stego samples of shape ``[num_stegos, num_subspace_dims]``.
        :param w: unsigned weights of shape ``[num_subspace_dims]``.
        :return: ``(w, bias)``, with ``w`` signed.
        """
        num_covers = len(Xc)
        num_stegos = len(Xs)

        y_pred_covers = Xc @ w
        y_pred_stegos = Xs @ w
        y_pred = np.concatenate([y_pred_covers, y_pred_stegos])

        y_true = np.concatenate([
            np.zeros(num_covers),
            np.ones(num_stegos),
        ])

        # Sort predictions
        permutation = np.argsort(y_pred)
        y_pred = y_pred[permutation]
        y_true = y_true[permutation]

        threshold_idx, sgn = self._search_threshold(y_true, num_covers, num_stegos)

        # Calculate bias term
        bias = sgn * 0.5 * (y_pred[threshold_idx] + y_pred[threshold_idx + 1])
        if sgn == -1:
            # Flip sign if needed
            w = -w

        return w, bias

    @staticmethod
    def _search_threshold(
        y_true: np.ndarray,
        num_covers: int,
        num_stegos: int,
    ) -> typing.Tuple[typing.Optional[int], typing.Optional[int]]:
        """Search the best threshold over the sorted samples, moving from low to high scores.

        Case 1 (``sgn = 1``): covers score lower than stegos; below the threshold is predicted
        cover, above stego. Case 2 (``sgn = -1``): the opposite. With ``c`` and ``s`` the numbers
        of covers and stegos among the samples below the threshold, case 1 has error
        ``num_covers - c + s`` and case 2 ``num_stegos + c - s``. The first minimum in the order
        ``error1(0), error2(0), error1(1), ...`` wins, and only if it beats ``num_covers``, which
        is the error of the lowest possible threshold. This is the same selection, including
        ties, as a loop over the samples that keeps a running minimum with a strict ``<``.

        :param y_true: labels (0 cover, 1 stego) sorted by score, of shape ``[num_samples]``.
        :param num_covers: number of covers.
        :param num_stegos: number of stegos.
        :return: ``(threshold_idx, sgn)``; the threshold lies between sorted samples
            ``threshold_idx`` and ``threshold_idx + 1``.
        """
        num_samples = num_covers + num_stegos

        is_cover = y_true[:-1] == 0
        c = np.cumsum(is_cover)
        s = np.cumsum(~is_cover)

        error1 = num_covers - c + s
        error2 = num_stegos + c - s

        # Interleave as [error1(0), error2(0), error1(1), error2(1), ...]
        candidates = np.empty(2 * (num_samples - 1), dtype=error1.dtype)
        candidates[0::2] = error1
        candidates[1::2] = error2

        best = int(np.argmin(candidates))
        if candidates[best] < num_covers:  # initial E_min in the loop
            return best // 2, (1 if best % 2 == 0 else -1)
        return None, None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scores; positive means stego.

        :param X: samples of shape ``[num_samples, num_features]``.
        :return: scores of shape ``[num_samples]``.
        :raises AttributeError: if the learner is not fitted.
        """
        if self.w is None or self.b is None:
            raise AttributeError("Weights or bias not initialized yet. Have you trained the classifier?")

        return X @ self.w - self.b

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate the accuracy.

        :param X: samples of shape ``[num_samples, num_features]``.
        :param y_true: labels of shape ``[num_samples]``, 0 for covers and 1 for stegos.
        :return: accuracy.
        :raises ValueError: if ``y_true`` does not contain exactly the labels 0 and 1.
        """
        if set(np.unique(y_true)) != {0, 1}:
            raise ValueError("Expected input labels with values 0 and 1")

        y_pred = (self.predict(X) > 0).astype(int)
        return (y_true == y_pred).mean()
