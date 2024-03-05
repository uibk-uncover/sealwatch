import numpy as np


class OutOfBagErrorEstimates(object):
    def __init__(self, num_covers, num_stegos):
        """
        Initialize out-of-bag error estimates object.
        The name out-of-bag estimate comes from bagging (bootstrap aggregation), which is an established technique for reducing the variance of classifiers.

        :param num_covers: number of cover images
        :param num_stegos: number of stego images
        """
        self.num_covers = num_covers
        self.num_stegos = num_stegos

        # While adding base learners to the ensemble, keep track of the average prediction
        self.Xc_fusion_majority_vote = np.zeros(num_covers, dtype=int)
        self.Xs_fusion_majority_vote = np.zeros(num_stegos, dtype=int)

        # Count how many times each sample was included in the bootstrap test samples
        self.Xc_num = np.zeros(num_covers, dtype=int)
        self.Xs_num = np.zeros(num_stegos, dtype=int)

        # Set up random number generator for breaking ties. This is useful for reproducibility.
        self.rng_for_ties = np.random.RandomState(1)

        # Keep track of statistics:
        # Average L in OOB
        self.xs = []

        # History of OOB errors
        self.ys = []

        # Current OOB error
        self.oob_error = None
