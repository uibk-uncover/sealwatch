
from parameterized import parameterized
import numpy as np
import scipy.io
# from scipy.io import loadmat
import sealwatch as sw
# from sealwatch.ensemble_classifier.fld_ensemble_trainer import FldEnsembleTrainer
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'spam'


class TestFldEnsembleClassifier(unittest.TestCase):

    @parameterized.expand([
        ("ensemble_matlab/tutorial_seed_12345.mat", 12345), # Tutorial, seed 12345
        ("ensemble_matlab/tutorial_seed_98765.mat", 98765),  # Tutorial, seed 98765
        # ("ensemble_matlab/gfr_qf95.mat", 12345),  # QF95, J-UNIWARD 0.4 bpnzAC stego, GFR features, first 2000 samples
    ])
    def test_tutorial_matlab_seed(self, matlab_trained_ensemble, seed):
        mat = scipy.io.loadmat(defs.ASSETS_DIR / matlab_trained_ensemble)

        Xc_train = np.ascontiguousarray(mat["TRN_cover"])
        Xs_train = np.ascontiguousarray(mat["TRN_stego"])

        trainer = sw.ensemble_classifier.FldEnsembleTrainer(
            Xc=Xc_train,
            Xs=Xs_train,
            seed=seed,
            verbose=0,
        )

        ensemble_classifier, training_records = trainer.train()

        for training_step, training_record in enumerate(training_records):
            # Our training results
            d_sub = training_record["d_sub"]
            num_base_learners = training_record["num_base_learners"]
            oob_error = training_record["oob_error"]

            # Compare to Matlab training results
            self.assertTrue(d_sub == mat["search_d_sub"].flatten()[training_step])
            self.assertTrue(num_base_learners == mat["search_L"].flatten()[training_step])
            self.assertTrue(np.isclose(oob_error, mat["search_oob"].flatten()[training_step]))


__all__ = ["TestFldEnsembleClassifier"]
