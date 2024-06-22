
import numpy as np


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
        Fit the classifier
        :param X: ndarray of shape [num_samples, num_features]
        :param y: ndarray of target labels, where -1 denotes the negative class and +1 denotes the positive class
        """
        # Validate input args
        assert set(np.unique(y)) == {-1, +1}, "Expected samples with -1 and +1 labels"

        # Split into covers and stegos
        cover_mask = (y == -1)
        stego_mask = (y == +1)

        Xc = X[cover_mask]
        Xs = X[stego_mask]

        num_covers = len(Xc)
        num_stegos = len(Xs)

        # Remove feature dimensions columns with constant values
        num_feature_dims = X.shape[1]
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

        # Within-class scatter matrix
        sigma_cs = sigma_c + sigma_s

        # Add stabilizing constant to ensure that the within-class scatter matrix is positive definite
        sigma_cs = sigma_cs + 1e-10 * np.eye(num_feature_dims)

        # Check for NaN values (may occur when the feature value is constant over images)
        drop_feature_dims = drop_feature_dims | np.any(np.isnan(sigma_cs), axis=0)

        # Drop feature dimensions from mean and covariance
        sigma_cs = sigma_cs[~drop_feature_dims, :][:, ~drop_feature_dims]
        mu = mu[~drop_feature_dims]

        # Calculate weights
        solved = False
        solve_counter = 0
        while not solved:
            try:
                # According to [Matlab's mldivide](https://de.mathworks.com/help/matlab/ref/mldivide.html), Matlab solves this linear system of equations via its Cholesky decomposition.
                w = np.linalg.solve(sigma_cs, mu)
                solved = True
            except np.linalg.LinAlgError:
                # Catch warnings about singular matrix

                # Increase regularization
                if 0 == solve_counter:
                    solve_counter = 1

                else:
                    solve_counter *= 5

                # Distance from 1 to the next larger representable real number in double precision
                eps = np.spacing(1)

                # Dynamically increase stabilizing constant
                sigma_cs += solve_counter * eps * np.eye(num_feature_dims)

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

        # Case 1: Covers received lower score than stego images (sgn = 1)
        # We start with the lowest possible threshold. At this threshold, there are no missed detections and all covers are misclassified as stego (false alarms).
        # Scores below the threshold are predicted as covers, while scores above the threshold are predicted as stego.
        MD = 0
        FA = num_covers
        error_per_threshold = np.zeros(num_samples - 1)

        # Case 2: Covers received higher score than stego images (sgn = -1)
        # Scores below the threshold are predicted as stego, while scores above the threshold are predicted as covers.
        MD2 = num_stegos
        FA2 = 0
        error_per_threshold2 = np.zeros(num_samples - 1)

        # Keep track of best threshold
        E_min = (FA + MD)
        threshold_idx = None
        sgn = None

        # Iterate over sorted probabilities, moving from low to high prediction scores
        for idx in range(num_samples - 1):
            if y_true[idx] == -1:
                # We encountered a cover
                # Case 1: Decrease false alarms
                FA = FA - 1
                # Case 2: Increase missed detections
                MD2 = MD2 + 1
            else:
                # We encountered a stego
                # Case 1: Increase missed detections
                MD = MD + 1
                # Case 2: Decrease false alarms
                FA2 = FA2 - 1

            # Recompute current error
            error_per_threshold[idx] = FA + MD
            error_per_threshold2[idx] = FA2 + MD2

            # Update optimal threshold
            if error_per_threshold[idx] < E_min:
                # Case 1
                E_min = error_per_threshold[idx]
                threshold_idx = idx
                sgn = 1

            if error_per_threshold2[idx] < E_min:
                # Case 2
                E_min = error_per_threshold2[idx]
                threshold_idx = idx
                sgn = -1

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
