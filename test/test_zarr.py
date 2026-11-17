"""Tests of the zarr feature store, :mod:`sealwatch.tools._zarr`, and of the loaders of paired features."""

import concurrent.futures
import pathlib
import tempfile
import unittest

import h5py
import numpy as np
import zarr

import sealwatch as sw

NUM_SAMPLES, NUM_FEATURES, PER_TASK = 95, 30, 20  # 5 tasks, the last one partial


def _features(rows):
    """Deterministic stand-in features of the given rows."""
    return np.random.default_rng(rows.start).random((len(rows), NUM_FEATURES), dtype=np.float32)


def _run_task(store, task):
    """One task of the job array, as a separate process would run it."""
    sw.tools.write_feature_task(store, task, _features(sw.tools.feature_task_rows(store, task)))


class TestFeatureStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = pathlib.Path(self._tmp.name) / "srm.zarr"
        self.filenames = [f"{i}.png" for i in range(NUM_SAMPLES)]
        sw.tools.create_feature_store(self.store, self.filenames, NUM_FEATURES, PER_TASK, samples_per_chunk=5)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tasks_split_the_rows(self):
        self.assertEqual(sw.tools.num_feature_tasks(self.store), 5)
        self.assertEqual(sw.tools.feature_task_rows(self.store, 0), range(0, 20))
        self.assertEqual(sw.tools.feature_task_rows(self.store, 4), range(80, 95))
        with self.assertRaises(IndexError):
            sw.tools.feature_task_rows(self.store, 5)

    def test_concurrent_tasks_and_missing(self):
        """Tasks in separate processes at the same time, one of them failed and rerun;
        the store then reads as one array, with one file per task."""
        with concurrent.futures.ProcessPoolExecutor(4) as pool:
            list(pool.map(_run_task, [self.store] * 4, [0, 1, 3, 4]))  # task 2 "failed"
        np.testing.assert_array_equal(sw.tools.missing_feature_tasks(self.store), [2])

        _run_task(self.store, 2)  # rerun
        self.assertEqual(len(sw.tools.missing_feature_tasks(self.store)), 0)
        self.assertEqual(sorted(p.name for p in self.store.iterdir()),
                         ["c.0.0", "c.1.0", "c.2.0", "c.3.0", "c.4.0", "zarr.json"])  # flat, no subfolders

        z = zarr.open_array(self.store, mode="r")
        expected = np.concatenate([_features(range(s, min(s + PER_TASK, NUM_SAMPLES)))
                                   for s in range(0, NUM_SAMPLES, PER_TASK)])
        np.testing.assert_array_equal(z[:], expected)
        np.testing.assert_array_equal(z[[3, 50, 94]], expected[[3, 50, 94]])  # random access
        self.assertEqual(z.attrs["filenames"], self.filenames)

    def test_invalid_writes_rejected(self):
        rows = sw.tools.feature_task_rows(self.store, 0)
        with self.assertRaises(ValueError):
            sw.tools.write_feature_task(self.store, 0, np.zeros((len(rows) - 1, NUM_FEATURES)))  # shape
        nan = np.zeros((len(rows), NUM_FEATURES))
        nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            sw.tools.write_feature_task(self.store, 0, nan)  # NaN marks missing rows

    def test_truncated_shard_is_missing(self):
        """A task killed while writing (e.g., at the time limit) leaves a cut shard file."""
        _run_task(self.store, 0)
        _run_task(self.store, 1)
        shard = self.store / "c.1.0"
        shard.write_bytes(shard.read_bytes()[:100])
        self.assertEqual(sw.tools.missing_feature_tasks(self.store).tolist(), [1, 2, 3, 4])

    def test_existing_store_not_overwritten(self):
        _run_task(self.store, 0)
        with self.assertRaises(Exception):
            sw.tools.create_feature_store(self.store, self.filenames, NUM_FEATURES, PER_TASK)
        self.assertEqual(sw.tools.missing_feature_tasks(self.store).tolist(), [1, 2, 3, 4])

    def test_load_like_hdf5(self):
        """The loaders take a zarr store as an HDF5 file: same features, filenames, and pairing.
        A missing task drops its pairs, as a failed extraction (all-zero rows) does in HDF5."""
        root = self.store.parent
        rng = np.random.default_rng(0)
        cover = rng.random((NUM_SAMPLES, NUM_FEATURES), dtype=np.float32)
        stego = cover + 0.1
        stego[:PER_TASK] = 0  # task 0 of the stegos failed: zero in HDF5, missing in zarr
        for name, features in (("cover", cover), ("stego", stego)):
            with h5py.File(root / f"{name}.h5", "w") as f:
                f["features"] = features
                f["filenames"] = np.array(self.filenames, dtype="S")
            store = root / f"{name}.zarr"
            sw.tools.create_feature_store(store, self.filenames, NUM_FEATURES, PER_TASK)
            for task in range(sw.tools.num_feature_tasks(store)):
                rows = sw.tools.feature_task_rows(store, task)
                if not (name == "stego" and task == 0):
                    sw.tools.write_feature_task(store, task, features[rows.start:rows.stop])

        features, filenames = sw.tools.load_zarr(root / "cover.zarr", max_num_samples=10)
        np.testing.assert_array_equal(features, cover[:10])
        self.assertEqual(filenames.tolist(), self.filenames[:10])

        from_hdf5 = sw.tools.load_features(root / "cover.h5", root / "stego.h5")
        from_zarr = sw.tools.load_features(root / "cover.zarr", root / "stego.zarr")
        self.assertEqual(len(from_zarr[0]), NUM_SAMPLES - PER_TASK)
        for a, b in zip(from_hdf5, from_zarr):
            np.testing.assert_array_equal(a, b)

        # the old location still works, with a warning
        with self.assertWarns(DeprecationWarning):
            from_helpers = sw.ensemble_classifier.helpers.load_features(root / "cover.h5", root / "stego.h5")
        for a, b in zip(from_hdf5, from_helpers):
            np.testing.assert_array_equal(a, b)


__all__ = ["TestFeatureStore"]
