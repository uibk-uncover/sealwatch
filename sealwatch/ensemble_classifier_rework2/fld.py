"""Fisher linear discriminant (FLD) base learner for binary steganalysis.

This module provides :class:`FisherLinearDiscriminantLearner`, a single FLD
classifier whose projection direction is the solution to the generalised
eigenvalue problem on the within-class scatter matrix (solved via Cholesky
decomposition for speed). The decision threshold is set to minimise the
total detection error ``P_E`` under equal priors.

Compared to the legacy implementation this version adds:

* :meth:`~FisherLinearDiscriminantLearner.fit_presplit` -- fits directly on
  pre-split cover/stego arrays without concat + re-split (halves per-learner
  memory transient).
* A vectorized threshold sweep in :meth:`~FisherLinearDiscriminantLearner._find_threshold`
  (cumsum instead of a Python loop; ~4x faster, bit-identical).
* A dtype-aware ridge for the scatter matrix (safe for both float64 and float32).
* Cholesky solve (``scipy.linalg.solve(..., assume_a="pos")``) instead of LU
  (~2x faster on symmetric positive-definite systems).
"""

import numpy as np
import scipy.linalg


class FisherLinearDiscriminantLearner(object):
    def __init__(self, w=None, b=None):
        """
        Initialize Fisher linear discriminant (FLD) base learner.
        The decision threshold of each base learner is adjusted to minimize the total detector error under equal priors on the training set
            P_E = min_{P_FA} 1/2 ( P_FA + P_MD(P_FA) ) ,
        where P_FA is the probability of false alarms and P_MD is the probability of missed detections.

        Advantages of FLD classifiers:
        - Low training complexity
        - A single FLD classifier is relatively weak and unstable, but forming an ensemble of FLD classifiers increases diversity.
        """
        self.w = w
        self.b = b

    def fit(self, X, y):
        """
        Fit the classifier on a stacked design matrix.
        :param X: ndarray of shape [num_samples, num_features]
        :param y: target labels, where -1 denotes the negative (cover) class and +1 the positive (stego) class
        """
        assert set(np.unique(y)) == {-1, +1}, "Expected samples with -1 and +1 labels"
        return self.fit_presplit(X[y == -1], X[y == +1])

    def fit_presplit(self, Xc, Xs):
        """
        Fit directly on pre-split cover/stego blocks -- i.e. without concatenating them
        into one matrix and boolean-indexing it back apart.

        BaseLearner already holds the projected (bootstrap x subspace) cover and stego
        data as two separate arrays. Routing them straight in here avoids materializing
        that projected data two extra times per base learner (the concatenate, then the
        boolean re-split) -- a pure memory/transient saving; the results are identical.

        :param Xc: cover samples of shape [num_covers, num_features]
        :param Xs: stego samples of shape [num_stegos, num_features]
        """
        num_covers = len(Xc)
        num_stegos = len(Xs)

        # Remove feature dimensions columns with constant values
        num_feature_dims = Xc.shape[1]
        drop_feature_dims = np.zeros(num_feature_dims, dtype=bool)

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

        # Calculate means of each class
        mu_c = np.mean(Xc, axis=0)
        mu_s = np.mean(Xs, axis=0)
        mu = (mu_s - mu_c).T

        # Calculate covariance for covers
        Xc_zero_mean = Xc - mu_c[None, :]
        sigma_c = Xc_zero_mean.T @ Xc_zero_mean
        sigma_c /= num_covers

        # Calculate covariance for stegos
        Xs_zero_mean = Xs - mu_s[None, :]
        sigma_s = Xs_zero_mean.T @ Xs_zero_mean
        sigma_s /= num_stegos

        # Within-class scatter matrix (fresh array -> safe to mutate in place)
        sigma_cs = sigma_c + sigma_s

        # Stabilize the (symmetric PSD) scatter matrix by adding a small ridge to its
        # diagonal so it is positive definite for the Cholesky solve. Done in place on
        # the diagonal -- equivalent to sigma_cs + ridge*np.eye(d) but without
        # allocating a dxd identity and without upcasting a float32 matrix to float64.
        if sigma_cs.dtype == np.float64:
            # Legacy constant -> bit-identical to the Matlab reference in compat mode.
            ridge = 1e-10
        else:
            # In lower precision (e.g. float32) the 1e-10 constant is far below the
            # machine epsilon (~1.2e-7 for float32), so it does not regularize an
            # ill-conditioned scatter matrix at all -- the float32 Cholesky then
            # generates denormals and a ~10x slowdown. Use a scale-aware ridge (~10x
            # the float32 epsilon, relative to the mean diagonal) so float32 stays both
            # stable and fast across all subspace sizes.
            ridge = 1e-6 * (np.trace(sigma_cs) / num_feature_dims)
        sigma_cs.flat[::num_feature_dims + 1] += ridge

        # Check for NaN values (may occur when the feature value is constant over images)
        drop_feature_dims = drop_feature_dims | np.any(np.isnan(sigma_cs), axis=0)

        # Drop feature dimensions from mean and covariance
        sigma_cs = sigma_cs[~drop_feature_dims, :][:, ~drop_feature_dims]
        mu = mu[~drop_feature_dims]

        # Calculate weights.
        # sigma_cs is symmetric positive definite (sum of covariance matrices plus a
        # stabilizing epsilon*I), so solve via Cholesky (assume_a="pos"). This is
        # ~2x faster than a general LU solve and matches Matlab's mldivide, which
        # also uses Cholesky for SPD systems.
        solved = False
        solve_counter = 0
        while not solved:
            try:
                w = scipy.linalg.solve(sigma_cs, mu, assume_a="pos")
                solved = True
            except np.linalg.LinAlgError:
                # Numerically not positive definite / singular -> add regularization
                if 0 == solve_counter:
                    solve_counter = 1
                else:
                    solve_counter *= 5
                eps = np.spacing(1)
                sigma_cs += solve_counter * eps * np.eye(len(sigma_cs))

        if len(sigma_cs) != len(sigma_c):
            # Resolve previously found NaN columns: Set the corresponding elements of w equal to zero
            w_new = np.zeros(num_feature_dims)
            w_new[~drop_feature_dims] = w
            w = w_new

        # Adjust threshold to minimize the total error under equal priors
        w, b = self._find_threshold(Xc, Xs, w)

        self.w = w
        self.b = b

    @staticmethod
    def _find_threshold(Xc, Xs, w):
        """
        Find threshold through minimizing (P_MD + P_FA) / 2, where P_MD stands for the missed detection rate and P_FA for the false alarm rate.
        :param Xc: cover samples of shape [num_cover_samples, num_subspace_dims]
        :param Xs: stego samples of shape [num_stego_samples, num_subspace_dims]
        :param w: unsigned base learner weights of shape [num_subspace_dims]
        :return: (w, bias) as 2-tuple
            w are the signed base learner weights. Compared to the given w argument, the sign could have switched.
            bias is a scalar
        """

        num_covers = len(Xc)
        num_stegos = len(Xs)
        num_samples = num_covers + num_stegos

        y_pred_covers = Xc @ w
        y_pred_stegos = Xs @ w
        y_pred = np.concatenate([y_pred_covers, y_pred_stegos])

        y_true = np.concatenate([
            -np.ones(num_covers),
            +np.ones(num_stegos),
        ])

        # Sort predictions
        permutation = np.argsort(y_pred)
        y_pred = y_pred[permutation]
        y_true = y_true[permutation]

        # The base learner only aimed to spread the probabilities, but did not take care of the correct class assignment.
        # This method automatically decides whether the sign of the weights needs to be flipped.
        #
        # Vectorized sweep over thresholds (replaces the original per-sample Python
        # loop; bit-identical selection). The threshold moves from low to high; at
        # position idx the samples 0..idx are "below" it. Let c[idx] / s[idx] be the
        # number of covers / stegos among the first idx+1 sorted samples.
        #
        # Case 1 (sgn=+1, covers score lower): FA = num_covers - c, MD = s
        #   -> error1 = num_covers - c + s
        # Case 2 (sgn=-1, covers score higher): FA2 = -s, MD2 = num_stegos + c
        #   -> error2 = num_stegos + c - s
        # The original loop runs idx in 0..num_samples-2 and uses strict "<" against a
        # running minimum that starts at num_covers, so the chosen (idx, sgn) is the
        # FIRST occurrence -- in the order e1(0), e2(0), e1(1), e2(1), ... -- of the
        # global minimum, and only if that minimum beats num_covers.
        is_cover = y_true[:-1] == -1
        c = np.cumsum(is_cover)
        s = np.cumsum(~is_cover)

        error1 = num_covers - c + s
        error2 = num_stegos + c - s

        # Interleave as [e1(0), e2(0), e1(1), e2(1), ...]; argmin returns the first
        # occurrence of the minimum, matching the loop's "case 1 before case 2,
        # earliest index wins" tie-breaking.
        candidates = np.empty(2 * (num_samples - 1), dtype=error1.dtype)
        candidates[0::2] = error1
        candidates[1::2] = error2

        best = int(np.argmin(candidates))
        if candidates[best] < num_covers:  # initial E_min in the legacy loop
            threshold_idx = best // 2
            sgn = 1 if best % 2 == 0 else -1
        else:
            threshold_idx = None
            sgn = None

        # Calculate bias term
        bias = sgn * 0.5 * (y_pred[threshold_idx] + y_pred[threshold_idx + 1])
        if sgn == -1:
            # Flip sign if needed
            w = -w

        return w, bias

    def predict(self, X):
        """
        Predict scores for given samples
        :param X: samples of shape [num_samples, num_features]
        :return: soft predictions of shape [num_samples]
        """
        if self.w is None or self.b is None:
            raise AttributeError("Weights or bias not initialized yet. Have you trained the classifier?")

        return X @ self.w - self.b

    def score(self, X, y_true):
        """
        Calculate accuracy
        :param X: samples of shape [num_samples, num_features]
        :param y_true: labels of shape [num_samples], where -1 indicates a cover image and +1 indicates a stego image
        :return: accuracy
        """
        assert set(np.unique(y_true)) == {-1, +1}, "Expected input labels with values -1 and +1"

        y_pred = np.sign(self.predict(X))
        return (y_true == y_pred).mean()
