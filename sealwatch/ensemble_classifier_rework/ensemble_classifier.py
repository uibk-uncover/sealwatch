import numpy as np
# from sklearn.metrics import accuracy_score


class EnsembleClassifier(object):
    def __init__(self, base_learners, d_sub=None):
        """
        Initialize an ensemble of base learners

        Use the FldEnsembleTrainer to obtain a trained ensemble classifier.

        :param base_learners: list of base learner objects
        :param d_sub: subspace dimensionality; not used, only for read access
        """
        self.base_learners = base_learners
        self._d_sub = d_sub

    @property
    def d_sub(self):
        return self._d_sub

    @property
    def num_base_learners(self):
        return len(self.base_learners)

    def predict_confidence(self, X):
        """
        Calculate confidence score based on majority voting.
        :param X: samples of shape [num_samples, num_features]
        :return: confidence score of predictions of shape [num_samples], in the range of -1 for the negative and +1 for the positive class.
        """
        y_conf = np.zeros(len(X), dtype=float)
        for base_learner in self.base_learners:
            y_conf += np.sign(base_learner.predict(X)).astype(int)

        return y_conf / len(self.base_learners)

    def predict(self, X):
        """
        Calculate predictions based on (unweighted) majority voting. Ties are resolved randomly.
        :param X: samples of shape [num_samples, num_features]
        :return: predictions of shape [num_samples], where -1 stands for the negative and +1 for the positive class
        """
        y_pred = np.zeros(len(X), dtype=int)
        for base_learner in self.base_learners:
            y_pred += np.sign(base_learner.predict(X)).astype(int)

        # Resolve ties
        rng_for_ties = np.random.RandomState(6020)

        # If tie - random
        tie_mask = y_pred == 0
        num_ties = np.sum(tie_mask)
        y_pred[tie_mask] = np.sign(rng_for_ties.rand(num_ties) - 0.5).astype(int)

        return np.sign(y_pred).astype(int)

    def score(self, X, y_true):
        """
        Calculate accuracy
        :param X: samples of shape [num_samples, num_features]
        :param y_true: labels of shape [num_samples], where -1 indicates a cover image and +1 indicates a stego image
        :return: accuracy
        """

        assert set(y_true).union({-1, +1}) == {
            -1,
            1,
        }, "Expected input labels with values -1 and +1"

        y_pred = self.predict(X)
        return (y_true == y_pred).mean()
        # return accuracy_score(y_true, y_pred)
