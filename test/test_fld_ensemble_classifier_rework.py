"""
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


__all__ = ["TestFldEnsembleClassifier"]"""

import importlib
import unittest

import numpy as np
import scipy.io
from parameterized import parameterized

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / "features_matlab" / "spam"

# Short key -> package under test, so the same Matlab reference runs against both
# the legacy and the reworked implementation.
MODULES = {
    "legacy": "sealwatch.ensemble_classifier",
    "rework": "sealwatch.ensemble_classifier_rework",
}


class TestFldEnsembleClassifier(unittest.TestCase):

    @parameterized.expand([
        ("tutorial_12345_legacy", "ensemble_matlab/tutorial_seed_12345.mat", 12345, "legacy"),
        ("tutorial_12345_rework", "ensemble_matlab/tutorial_seed_12345.mat", 12345, "rework"),
        ("tutorial_98765_legacy", "ensemble_matlab/tutorial_seed_98765.mat", 98765, "legacy"),
        ("tutorial_98765_rework", "ensemble_matlab/tutorial_seed_98765.mat", 98765, "rework"),
        # ("gfr_qf95_legacy", "ensemble_matlab/gfr_qf95.mat", 12345, "legacy"),
        # ("gfr_qf95_rework", "ensemble_matlab/gfr_qf95.mat", 12345, "rework"),
    ])
    def test_tutorial_matlab_seed(self, _name, matlab_trained_ensemble, seed, module_key):
        # Resolve the package lazily so the same test works under both the
        # lightweight conftest bootstrap and a full (Rust-built) install.
        module = importlib.import_module(MODULES[module_key])

        mat = scipy.io.loadmat(defs.ASSETS_DIR / matlab_trained_ensemble)
        Xc_train = np.ascontiguousarray(mat["TRN_cover"])
        Xs_train = np.ascontiguousarray(mat["TRN_stego"])

        trainer = module.FldEnsembleTrainer(
            Xc=Xc_train,
            Xs=Xs_train,
            seed=seed,
            verbose=0,
        )
        _ensemble, training_records = trainer.train()

        # Matlab reference trajectory
        search_d_sub = mat["search_d_sub"].flatten()
        search_L = mat["search_L"].flatten()
        search_oob = mat["search_oob"].flatten()

        self.assertEqual(
            len(training_records), len(search_d_sub),
            f"[{module_key}] number of search steps differs from Matlab")

        for step, record in enumerate(training_records):
            self.assertEqual(
                record["d_sub"], int(search_d_sub[step]),
                f"[{module_key}] d_sub mismatch at step {step}")
            self.assertEqual(
                record["num_base_learners"], int(search_L[step]),
                f"[{module_key}] L mismatch at step {step}")
            self.assertTrue(
                np.isclose(record["oob_error"], search_oob[step]),
                f"[{module_key}] OOB mismatch at step {step}: "
                f"{record['oob_error']} vs {search_oob[step]}")


__all__ = ["TestFldEnsembleClassifier"]
