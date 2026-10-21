"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import re
import scipy.io
import subprocess
import sealwatch as sw
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs

FEATURE_DIR = defs.ASSETS_DIR / 'features_matlab' / 'srm'
# FEATURE_DIR = defs.ASSETS_DIR / 'features_matlab' / 'srm'


class TestSRM(unittest.TestCase):
    """Test suite for srm module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract_q1(self, fname):
        """Default backend against the Matlab reference."""
        #
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        start = time.perf_counter()
        features = sw.srmq1.extract(x0)
        end = time.perf_counter()
        self._logger.info(f'extracting SRMQ1 from {fname}.jpeg took {end - start} s')

        #
        self.assertEqual(len(sw.tools.flatten(features)), 12753)  # from DDE page

        # compare to reference
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        features_ref = OrderedDict([(k, features_ref[k]) for k in features_ref.keys() if k.endswith('_q1')])

        for submodel, f in features.items():
            f_ref = features_ref[submodel].flatten()

            # Compare to Matlab features
            # print(submodel, np.allclose(f, f_ref))
            np.testing.assert_allclose(f, f_ref)

        # same order as Matlab
        self.assertEqual(list(features.keys()), list(features_ref.keys()))
        np.testing.assert_allclose(
            sw.tools.flatten(features),
            sw.tools.flatten(features_ref)
        )

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        """Default backend against the Matlab reference."""
        self._logger.info(f'TestSRM.test_extract({fname})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        start = time.perf_counter()
        features = sw.srm.extract(x0)
        end = time.perf_counter()
        self._logger.info(f'extracting SRM from {fname}.jpeg took {end - start} s')

        # check length
        self.assertEqual(len(sw.tools.flatten(features)), 34671)  # from DDE page

        # compare to Matlab reference
        features_ref = scipy.io.loadmat(FEATURE_DIR / f'{fname}.mat', simplify_cells=True)
        for submodel, f in features.items():
            f_ref = features_ref[submodel].flatten()

            # Compare to Matlab features
            # print(submodel, np.allclose(f, f_ref))
            np.testing.assert_allclose(f, f_ref)

        # same order as Matlab
        features_ref = OrderedDict([(k, v) for k, v in features_ref.items() if not k.startswith('__')])
        self.assertEqual(list(features.keys()), list(features_ref.keys()))
        np.testing.assert_allclose(
            sw.tools.flatten(features),
            sw.tools.flatten(features_ref)
        )


    @parameterized.expand([
        ([[2], [1.5, 1], [], [1], [2]],),  # unmerged spam11 of 3x3 and 5x5, missing family
        ([[2, 1]],),  # reversed order, shorter
        ([[1], [2.0], [1, 1.5], [1.5, 2], [2, 1.5]],),  # float step, merged in reversed order
    ])
    def test_order_qs(self, qs):
        """Custom quantization steps give the same submodels, ordered by family, then qs, then Matlab order."""
        self._logger.info(f'TestSRM.test_order_qs({qs=})')
        x = np.random.default_rng(12345).integers(0, 256, (40, 40)).astype(np.uint8)
        qs_copy = [list(q) for q in qs]
        features = sw.srm.extract(x, qs=qs)
        self.assertEqual(qs, qs_copy)  # parameter not modified
        # same submodels as computed one family and step at a time
        expected = {}
        for f, family in enumerate(sw.srm.srm.FAMILIES):
            for q in (qs[f] if f < len(qs) else []):
                single = [[]] * 5
                single[f] = [q]
                for name, v in sw.srm.extract(x, qs=single).items():
                    expected.setdefault(name, v)
        # spam11 of 3x3 and 5x5 with the same step are merged
        for name in [n for n in expected if n.startswith('s3x3_spam11_')]:
            name5 = name.replace('s3x3', 's5x5')
            if name5 in expected:
                expected[name.replace('s3x3', 's35')] = np.hstack([expected.pop(name), expected.pop(name5)])
        names = list(features.keys())
        self.assertEqual(set(names), set(expected))
        for name in names:
            np.testing.assert_array_equal(features[name], expected[name])
        # ordered by family, then by the position of the step in qs
        def position(name):
            family, _, q = name.split('_')
            f = sw.srm.srm.FAMILIES.index('s5x5' if family == 's35' else family)
            return f, [f'q{s}'.replace('.', '') for s in qs[f]].index(q)
        self.assertEqual([position(n) for n in names], sorted(position(n) for n in names))

    @parameterized.expand([['srm'], ['srmq1']])
    def test_extract_python(self, model):
        """Python implementation against the Matlab reference, on a single image."""
        self._logger.info(f'TestSRM.test_extract_python({model=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        with sw.BACKEND_PYTHON:
            features = getattr(sw, model).extract(x0)
        self.assertEqual(len(sw.tools.flatten(features)), {'srm': 34671, 'srmq1': 12753}[model])
        features_ref = scipy.io.loadmat(FEATURE_DIR / 'seal1.mat', simplify_cells=True)
        features_ref = OrderedDict([(k, v) for k, v in features_ref.items() if k in features])
        self.assertEqual(list(features.keys()), list(features_ref.keys()))
        for submodel, f in features.items():
            np.testing.assert_allclose(f, features_ref[submodel].flatten())

    @parameterized.expand([
        ('seal1', None),
        ('random', None),
        ('random', [[1]] * 5),  # SRMQ1
        ('random', [[2], [1.5, 1], [], [1], [2]]),  # unmerged spam11, missing family
        ('random', [[1], [2.0], [1, 1.5], [1.5, 2], [2, 1.5]]),  # float step, reversed order
        ('odd', None),
        ('flipped', None),  # not contiguous
    ])
    def test_rs_python(self, image, qs):
        """Rust implementation is bit-identical to the Python implementation."""
        self._logger.info(f'TestSRM.test_rs_python({image=}, {qs=})')
        rng = np.random.default_rng(12345)
        if image == 'seal1':
            x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        elif image == 'odd':
            x = rng.integers(0, 256, (37, 53)).astype(np.uint8)
        else:
            # smooth-ish, so that the residuals are not truncated everywhere
            x = np.clip(rng.normal(128, 40, (64, 64)).cumsum(axis=1) / np.arange(1, 65) + rng.normal(0, 3, (64, 64)), 0, 255).astype(np.uint8)
            if image == 'flipped':
                x = x[::-1, ::2]
        kw = {} if qs is None else {'qs': qs}
        with sw.BACKEND_PYTHON:
            f_py = sw.srm.extract(x, **kw)
        with sw.BACKEND_RUST:
            f_rs = sw.srm.extract(x, **kw)
        self.assertEqual(list(f_py.keys()), list(f_rs.keys()))
        for name in f_py:
            self.assertEqual(f_py[name].dtype, f_rs[name].dtype, name)
            np.testing.assert_array_equal(f_py[name], f_rs[name], err_msg=name)

    def test_rs_type(self):
        self._logger.info('TestSRM.test_rs_type()')
        with sw.BACKEND_RUST:
            with self.assertRaises(TypeError):
                sw.srm.extract(np.zeros((32, 32), dtype='float64'))

    @parameterized.expand([['BACKEND_PYTHON'], ['BACKEND_RUST']])
    def test_order_deterministic(self, backend):
        """The order of the submodels does not depend on the hash seed of the process."""
        self._logger.info(f'TestSRM.test_order_deterministic({backend=})')
        code = (
            'import numpy as np, sealwatch as sw; '
            'x = np.random.default_rng(0).integers(0, 256, (40, 40)).astype(np.uint8); '
            f'sw.set_backend(sw.{backend}); '
            'print(" ".join(sw.srm.extract(x).keys()))'
        )
        orders = {
            subprocess.run(
                [sys.executable, '-c', code],
                env={**os.environ, 'PYTHONHASHSEED': str(seed)},
                capture_output=True, text=True, check=True,
            ).stdout
            for seed in range(4)
        }
        self.assertEqual(len(orders), 1)


__all__ = ["TestSRM"]
