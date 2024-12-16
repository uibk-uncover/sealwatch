import numpy as np
import os
from scipy.signal import convolve

from .base_learner import BaseLearner
from .ensemble_classifier import EnsembleClassifier
from .out_of_bag_error_estimates import OutOfBagErrorEstimates
from .subspace_dimensionality_search import SubspaceDimensionalitySearch, FixedDimensionalityDummySearch
from .. import tools

# from sealwatch.utils.logger import setup_custom_logger
# from sealwatch.ensemble_classifier.base_learner import BaseLearner
# from sealwatch.ensemble_classifier.ensemble_classifier import EnsembleClassifier
# from sealwatch.ensemble_classifier.out_of_bag_error_estimates import OutOfBagErrorEstimates
# from sealwatch.ensemble_classifier.subspace_dimensionality_search import SubspaceDimensionalitySearch, FixedDimensionalityDummySearch
# from sealwatch.utils.matlab import randperm_naive


log = tools.setup_custom_logger(os.path.basename(__file__))


class FldEnsembleTrainer(object):
    def __init__(self, Xc, Xs, seed=None, seed_subspaces=None, seed_bootstrap=None, L="automatic", d_sub="automatic", verbose=1, max_num_base_learners=500):
        """
        Initialize a trainer for creating an ensemble classifier consisting of Fisher linear discriminant base learners.
        This implementation assumes pairs of cover and stego images. Hence, the number of samples in Xc and Xs must be identical.

        Setting the subspace dimensionality ``1d_sub``
        * If ``d_sub`` is set too low, the individual base learners will not be able to learn anything.
        * If ``d_sub`` is set too high, the individual base learners will become more dependent, which decreases the chance of learning non-linear classification boundaries. Furthermore, individual base learners may overfit when the number of training examples is relatively low compared to the number of feature dimensions.

        Setting the number of base learners ``L``
        * The ensemble benefits from more base learners until performance starts to saturate.
        * For automatic selection of ``L``, we observe the progress of the OOB estimate. Training terminates when the last 50 moving averages calculated from 5 consecutive OOB estimates lie in an epsilon-tube.

        Reference:
        J. Kodovský, J. Fridrich, and V. Holub, Ensemble Classifiers for Steganalysis of Digital Media. IEEE Transactions on Information Forensics and Security, Vol. 7, No. 2, pp. 432-444, April 2012.
        Paper: http://dde.binghamton.edu/download/ensemble/TIFS-2011-ensemble.pdf
        Webpage: http://dde.binghamton.edu/download/ensemble/

        :param Xc: cover training samples of shape [num_samples, num_features]
        :param Xs: stego training samples of shape [num_samples, num_features]; should have the same number of samples and dimensionality as the covers
        :param seed: seed for main random number generator
        :param seed_subspaces: seed for subspace random number generator
        :param seed_bootstrap: seed for bootstrap random number generator
        :param L: number of base learners; should be an integer; use "automatic" to determine this number automatically
        :param d_sub: subspace dimensionality; should be an integer; use "automatic" to find this number via a grid search
        :param verbose: value 1 prints log message; value 0 hides log messages
        :param max_num_base_learners: maximum number of base learners
        """
        # Sanity checks
        assert Xc.shape[1] == Xs.shape[1], "Expected cover and stego features to have the same dimensionality"
        assert Xc.shape[0] == Xs.shape[0], "Expected same number of cover and stego samples"
        assert isinstance(L, (int, np.integer)) or "automatic" == L, "Expected L to be an integer or the keyword \"automatic\""
        assert isinstance(d_sub, (int, np.integer)) or "automatic" == d_sub, "Expected d_sub to be an integer or the keyword \"automatic\""

        if not Xc.flags["C_CONTIGUOUS"] or not Xs.flags["C_CONTIGUOUS"]:
            log.warning("Training will be significantly faster when the input arrays are C-contiguous.")

        # Ensure double precision
        self.Xc = Xc.astype(np.float64)
        self.Xs = Xs.astype(np.float64)

        # Set up three Mersenne twister random number generators
        # (1) Main rng
        self.rng_main = np.random.RandomState(seed)

        # (2) Random subspace projection rng
        if seed_subspaces:
            self.seed_subspaces = seed_subspaces
        else:
            self.seed_subspaces = int(np.ceil(self.rng_main.random() * 1e9))

        self.rng_subspaces = np.random.RandomState(self.seed_subspaces)

        # (3) Bootstrap rng
        if seed_bootstrap:
            self.seed_bootstrap = seed_bootstrap
        else:
            self.seed_bootstrap = int(np.ceil(self.rng_main.random() * 1e9))

        self.rng_bootstrap = np.random.RandomState(self.seed_bootstrap)

        # Base learner settings
        self.L = L
        if self.L == "automatic":
            self.L_kernel = np.ones(5) / 5
            self.L_min_length = 25
            self.L_memory = 50
            self.L_epsilon = 0.005

        # Feature space settings
        self.d_sub = d_sub
        if self.d_sub == "automatic":
            self.oob_error_tolerance = 0.02
            self.initial_d_sub_step = 200
            self.search_for_d_sub = True
        else:
            self.search_for_d_sub = False

        # Maximum subspace size
        self.max_dim = Xc.shape[1]

        # Special handling for single feature dimension
        if self.max_dim == 1:
            self.d_sub = 1
            self.search_for_d_sub = False

        # Copy remaining arguments
        self.verbose = verbose
        self.max_num_base_learners = max_num_base_learners

        if self.verbose:
            self._initial_screen_output()

    def _initial_screen_output(self):
        num_covers = len(self.Xc)
        num_stegos = len(self.Xs)
        num_feature_dims = self.Xc.shape[1]
        num_training_samples = num_covers + num_stegos

        log.info("Ensemble classification")
        log.info(f"  - Training samples: {num_training_samples} (covers: {num_covers}, stegos: {num_stegos})")
        log.info(f"  - Feature space dimensionality: {num_feature_dims}")

        # Report ensemble settings
        if isinstance(self.L, int):
            log.info(f"  - L: {self.L}")
        else:
            log.info(f"  - L: {self.L} (min {self.L_min_length}, max {self.max_num_base_learners}, length {self.L_memory}, eps {self.L_epsilon:5f})")

        # Report feature space settings
        if isinstance(self.d_sub, str):
            log.info(f"  - d_sub: automatic (OOB error tolerance {self.oob_error_tolerance:4f}, initial step {self.initial_d_sub_step})")
        else:
            log.info(f"  - d_sub: {self.d_sub}")

        log.info(f"  - Seed 1 (subspaces): {self.seed_subspaces}")
        log.info(f"  - Seed 2 (bootstrap): {self.seed_bootstrap}")
        log.info("")

    def _has_next_base_learner(self, i, oob):
        """
        Decide whether to train another base learner.

        For automatic selection of L, this method observes the progress of the OOB estimate.
        Training terminates when the last 50 moving averages calculated from 5 consecutive OOB estimates lie in an epsilon-tube.

        :param i: current base learner index
        :param oob: out-of-bag error estimates object
        :return: False if any stopping criterion has been reached, True otherwise
        """

        if "automatic" == self.L:
            # Check criteria for early stopping
            if len(oob.ys) == 0:
                # No history
                # Not sure whether this case is ever selected
                return False

            if len(oob.ys) < self.L_min_length:
                # There is a history, but we are below the minimum number of base learners
                return True

            # Check the improvement over the last steps
            # ys is the history of OOB error estimates
            ys = oob.ys

            # Running average of length 5 over OOB error estimates
            # Only consider the latest L_memory steps
            # See Eq. 7
            A = convolve(ys[max(0, len(ys) - self.L_memory):], self.L_kernel, mode="valid")

            # Compare difference between minimum and maximum OOB error, as shown in Eq. 6.
            V = np.abs(np.max(A) - np.min(A))

            # If the moving average lies within an epsilon-tube, training terminates.
            if V < self.L_epsilon:
                return False

            # Have we exceeded the maximum number of base learners?
            if i == self.max_num_base_learners:
                if self.verbose:
                    log.info("Maximum number of base learners reached")
                return False

            # Otherwise, add another base learner
            return True

        else:
            # L was fixed
            # Check whether we have exceeded the desired number of base learners
            return i < self.L

    def _generate_random_subspace(self, d_sub):
        """
        Use the subspace rng to generate the next random subspace of length `d_sub`
        :return: ndarray of length `d_sub`. The array contains the subspace indices in range [0, `max_dim`)
        """

        subspace = tools.matlab.randperm_naive(self.rng_subspaces, self.max_dim)
        return subspace[:d_sub]

    def _regenerate_bootstrap_samples(self):
        """
        Use the bootstrap rng to split the samples into disjoint training and testing sets.
        The training samples are sampled with replacement, i.e., the same index can occur multiple times.

        According to the paper, roughly 63% of the samples are chosen for training, leaving 37% of samples to be used for validation.

        :return: (train_indices, test_indices)
        """
        num_covers = len(self.Xc)

        # Generate random indices between [0, num_covers) (with replacement). These will be used for training the base learner
        # Note that these indices can contain duplicates.
        train_indices = np.floor(num_covers * self.rng_bootstrap.rand(num_covers)).astype(int)

        # Get indices not included in `sub`
        # These will be used for testing.
        test_indices = np.setdiff1d(np.arange(num_covers), train_indices)

        return train_indices, test_indices

    def _update_oob_error_estimates(self, base_learner, test_indices, oob):
        # Update OOB error estimates
        # Predict bootstrap test samples
        Xc_proj = base_learner.predict(self.Xc, subset=test_indices)
        Xs_proj = base_learner.predict(self.Xs, subset=test_indices)

        # Keep track of how many times each sample included in the bootstrap test set
        oob.Xc_num[test_indices] += 1
        oob.Xs_num[test_indices] += 1

        # Average the predictions
        oob.Xc_fusion_majority_vote[test_indices] += np.sign(Xc_proj).astype(int)
        oob.Xs_fusion_majority_vote[test_indices] += np.sign(Xs_proj).astype(int)

        # Update errors
        num_covers = len(oob.Xc_fusion_majority_vote)
        num_stegos = len(oob.Xs_fusion_majority_vote)

        # tmp_c = np.copy(oob.Xc_fusion_majority_vote)
        # # Randomly nudge these samples where we are still undecided
        # undecided_mask = oob.Xc_fusion_majority_vote == 0
        # tmp_c[undecided_mask] = np.sign(oob.rng_for_ties.rand(np.sum(undecided_mask)) - 0.5)
        # Faster alternative:
        num_false_positives = np.sum(oob.Xc_fusion_majority_vote > 0)
        num_undecided = np.sum(oob.Xc_fusion_majority_vote == 0)
        num_false_positives += np.sum(oob.rng_for_ties.rand(num_undecided) > 0.5)

        # tmp_s = np.copy(oob.Xs_fusion_majority_vote)
        # # Randomly nudge these samples where we are still undecided
        # undecided_mask = oob.Xs_fusion_majority_vote == 0
        # tmp_s[undecided_mask] = np.sign(oob.rng_for_ties.rand(np.sum(undecided_mask)) - 0.5)
        # Faster alternative
        num_missed_detections = np.sum(oob.Xs_fusion_majority_vote < 0)
        num_undecided = np.sum(oob.Xs_fusion_majority_vote == 0)
        num_missed_detections += np.sum(oob.rng_for_ties.rand(num_undecided) < 0.5)

        # See Eq. 5.
        # oob_error = (np.sum(tmp_c > 0) + np.sum(tmp_s < 0)) / (num_covers + num_stegos)
        oob_error = (num_false_positives + num_missed_detections) / (num_covers + num_stegos)

        # The following result is not used anyway, so we can skip the computation.
        # Histogram could be replaced by bincount
        # data = np.concatenate([oob.Xc_num, oob.Xs_num])
        # # + 1 because we want to exceed the endpoint, and another + 1 because arange's upper bound is exclusive
        # bin_edges = np.arange(0, np.max(data) + 2) - 0.5
        # H, _ = np.histogram(
        #     a=data,
        #     bins=bin_edges,
        # )
        #
        # # Average L in OOB
        # avg_L = np.sum(H * np.arange(len(H))) / np.sum(H)

        # Keep track of statistics
        # oob.xs.append(avg_L)
        oob.ys.append(oob_error)

        return oob

    def train(self):
        """
        Train an ensemble of Fisher linear discriminant classifiers
        :return: (ensemble_classifier, training_records) as 2-tuple
        ensemble_classifier is an instance of an EnsembleClassifier
        training_records is a list of dicts
        """

        # Initialize the search for d_sub
        if self.search_for_d_sub:
            search = SubspaceDimensionalitySearch(
                d_sub_step=self.initial_d_sub_step,
                max_dim=self.max_dim,
                oob_error_tolerance=self.oob_error_tolerance)
        else:
            # Initialize dummy search object so that we omit checks for a non-null search object
            search = FixedDimensionalityDummySearch(d_sub=self.d_sub)

        # Keep track of currently best model
        min_oob_error = 1
        optimal_L = None
        optimal_d_sub = None
        trained_ensemble = None

        # Keep training logs for debugging and comparison against Matlab
        training_records = []

        # Outer loop optimizes for d_sub
        while search.search_in_progress:
            # Initialization
            num_base_learners = 0
            has_next_base_learner = True
            d_sub = search.next_d_sub
            base_learners = []
            oob = OutOfBagErrorEstimates(num_covers=len(self.Xc), num_stegos=len(self.Xs))

            # Inner loop optimizes for L
            while has_next_base_learner:
                # Random subspace generation
                subspace = self._generate_random_subspace(d_sub=d_sub)

                # Bootstrap initialization
                bootstrap_train_indices, bootstrap_test_indices = self._regenerate_bootstrap_samples()

                # Bootstrap: Select subset of samples for training
                # Bootstrapping is done inside the base learner, because it can implement the array access faster.
                # If you are unsure, you can also select the bootstrap samples beforehand as follows:
                # Xc_train = self.Xc[bootstrap_train_indices]
                # Xs_train = self.Xs[bootstrap_train_indices]

                # Training phase
                base_learner = BaseLearner()
                base_learner.fit(
                    Xc=self.Xc,
                    Xs=self.Xs,
                    subspace=subspace,
                    subset=bootstrap_train_indices,
                )

                base_learners.append(base_learner)

                # Increment base learner counter
                num_base_learners += 1

                # OOB error estimation
                oob = self._update_oob_error_estimates(base_learner, bootstrap_test_indices, oob)

                # Screen output
                if self.verbose:
                    log.info(f"  - d_sub {d_sub}, OOB {oob.ys[-1]:.4f}, L {num_base_learners}")

                has_next_base_learner = self._has_next_base_learner(num_base_learners, oob)

            # End while has next_base_learner

            # Update currently best model
            oob_error = oob.ys[-1]
            if oob_error < min_oob_error:
                min_oob_error = oob_error
                optimal_L = num_base_learners
                optimal_d_sub = d_sub
                trained_ensemble = base_learners

            # Update search
            search.update(d_sub, oob_error)

            # Keep log of results
            training_records.append({
                "d_sub": d_sub,
                "oob_error": oob_error,
                "num_base_learners": num_base_learners
            })

            # Screen output
            if self.verbose:
                # Empty line
                log.info("")

            # Clean up current iteration
            del base_learners
            del oob

        # Combine base learners into ensemble classifier
        ensemble_classifier = EnsembleClassifier(trained_ensemble, d_sub=optimal_d_sub)

        return ensemble_classifier, training_records
