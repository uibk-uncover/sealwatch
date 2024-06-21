import numpy as np
from sealwatch.ensemble_classifier.fld import FisherLinearDiscriminantLearner


class BaseLearner(object):
    def __init__(self):
        """
        Wrap a Fisher linear discriminant classifier that is trained with a given feature subspace
        """
        self.subspace = None
        self.learner = None

    @staticmethod
    def _fast_fancy_indexing(A, rows, cols):
        """
        Take a subset of a 2D array, but in a faster way than A[rows,: ][:, cols].
        See https://stackoverflow.com/questions/14386822/fast-numpy-fancy-indexing
        :param A: ndarray
        :param rows: row indices
        :param cols: column indices
        :return: A[rows, :][:, cols]
        """
        return (A.ravel()[(
                cols + (rows * A.shape[1]).reshape((-1, 1))
        ).ravel()]).reshape(rows.size, cols.size)

    def fit(self, Xc, Xs, subspace=None, subset=None):
        """
        Fit Fisher linear discriminant classifier on given samples
        :param Xc: cover features of shape [num_samples, num_features]
        :param Xs: stego features of shape [num_samples, num_features]
        :param subspace: ndarray containing the indices of the feature dimensions to retain
        :param subset: ndarray containing row indices, e.g., bootstrap training examples. Bootstrapping samples with replacement. Thus, the same index may appear multiple times.
        """

        # If subspace and subset are given, use fast fancy indexing
        # Although less readable than A[subset, :][:, subspace], this method is significantly faster.
        if subspace is not None and subset is not None:
            Xc = self._fast_fancy_indexing(Xc, cols=subspace, rows=subset)
            Xs = self._fast_fancy_indexing(Xs, cols=subspace, rows=subset)

        # Otherwise, use the traditional fancy indexing
        elif subspace is not None:
            Xc = Xc[:, subspace]
            Xs = Xs[:, subspace]

        elif subset is not None:
            Xc = Xc[subset]
            Xs = Xs[subset]

        # Set up labels
        yc = -np.ones(len(Xc), dtype=int)
        ys = +np.ones(len(Xs), dtype=int)

        # Concatenate covers and stegos to match common sklearn interface
        X = np.concatenate([Xc, Xs], axis=0)
        y = np.concatenate([yc, ys], axis=0)

        self.learner = FisherLinearDiscriminantLearner()
        self.learner.fit(X, y)
        self.subspace = subspace

    def predict(self, X, subset=None):
        """
        Obtain predictions from given set of samples
        :param X: test features of shape [num_samples, num_features]
        :param subset: ndarray containing row indices
        :return: predictions
        """

        # If subspace and subset are given, use fast fancy indexing
        if self.subspace is not None and subset is not None:
            X = self._fast_fancy_indexing(X, cols=self.subspace, rows=subset)

        # Otherwise, use the traditional fancy indexing
        elif self.subspace is not None:
            X = X[:, self.subspace]

        elif subset is not None:
            X = X[subset]

        return self.learner.predict(X)
