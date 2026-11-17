"""Regenerate reference.npz: outputs of the ensemble classifier of release 2026.11.

The tests check that FLD_LEGACY reproduces them bit for bit. The file is not meant to change;
regenerate it only to verify it. Run from the repository root:

    git worktree add --detach /tmp/sealwatch-2026.11 2026.11
    python test/assets/ensemble/numpy/make_reference.py /tmp/sealwatch-2026.11
    git worktree remove /tmp/sealwatch-2026.11

Contents, for each seed of the Matlab tutorial data in ../matlab (12345, 98765):
    tutorial_<seed>_records      d_sub, number of base learners, OOB error of each search step
    tutorial_<seed>_subspaces    subspace of each base learner
    tutorial_<seed>_w, _b        weights and threshold of each base learner
    tutorial_<seed>_confidence   confidence on the test samples (covers, then stegos)
    tutorial_<seed>_predict      prediction on the test samples
and of a single FLD on 200 small random datasets with many tied scores (10 to 59 samples per class):
    fld_w, fld_b                 weights and threshold of each
"""

import pathlib
import sys

import numpy as np
import scipy.io

HERE = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, sys.argv[1])  # the old release first, before an installed sealwatch
from sealwatch.ensemble_classifier import FldEnsembleTrainer  # noqa: E402
from sealwatch.ensemble_classifier.fld import FisherLinearDiscriminantLearner  # noqa: E402

out = {}
for seed in (12345, 98765):
    mat = scipy.io.loadmat(HERE.parent / 'matlab' / f'tutorial_seed_{seed}.mat')
    ensemble, records = FldEnsembleTrainer(
        Xc=np.ascontiguousarray(mat['TRN_cover']), Xs=np.ascontiguousarray(mat['TRN_stego']),
        seed=seed, verbose=0).train()
    X_test = np.concatenate([mat['TST_cover'], mat['TST_stego']])
    out[f'tutorial_{seed}_records'] = np.array([[r['d_sub'], r['num_base_learners'], r['oob_error']] for r in records])
    out[f'tutorial_{seed}_confidence'] = ensemble.predict_confidence(X_test)
    out[f'tutorial_{seed}_predict'] = ensemble.predict(X_test)
    out[f'tutorial_{seed}_subspaces'] = np.array([b.subspace for b in ensemble.base_learners])
    out[f'tutorial_{seed}_w'] = np.array([b.learner.w for b in ensemble.base_learners])
    out[f'tutorial_{seed}_b'] = np.array([b.learner.b for b in ensemble.base_learners])

# the same random datasets as in test_fld_matches_legacy_package
rng = np.random.RandomState(0)
w, b = [], []
for _ in range(200):
    n = rng.randint(10, 60)  # enough samples for a well-conditioned 5 x 5 scatter matrix
    X = np.round(rng.randn(2 * n, 5), rng.randint(0, 2))  # coarse values, many ties
    y = np.concatenate([-np.ones(n, dtype=int), np.ones(n, dtype=int)])
    X[n:, 0] += rng.rand()
    fld = FisherLinearDiscriminantLearner()
    fld.fit(X, y)
    w.append(fld.w)
    b.append(fld.b)
out['fld_w'], out['fld_b'] = np.array(w), np.array(b)

np.savez_compressed(HERE / 'reference.npz', **out)
print(f'Written {HERE / "reference.npz"}')
