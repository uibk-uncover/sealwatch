
import conseal as cl
import importlib.util
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import torch
import unittest
import urllib.request

from . import defs


REFERENCE_URL = 'https://raw.githubusercontent.com/revere7/UCNet_Steganalysis/098a557303554c724052d415565863ca99e3c29b'


def load_reference(tmpdir: str):
    """Downloads and imports UCNet implementation by the authors."""
    urllib.request.urlretrieve(f'{REFERENCE_URL}/UCNet_Spatial.py', os.path.join(tmpdir, 'UCNet_Spatial.py'))
    # imported under a different name
    urllib.request.urlretrieve(f'{REFERENCE_URL}/High-pass%20filters/srm_kernel_filters.py', os.path.join(tmpdir, 'srm_filter_kernel.py'))
    sys.path.insert(0, tmpdir)
    try:
        spec = importlib.util.spec_from_file_location('reference_ucnet', os.path.join(tmpdir, 'UCNet_Spatial.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(tmpdir)
        sys.modules.pop('srm_filter_kernel', None)
    return module


def to_reference(state_dict):
    """Converts state dict of UcNet to the reference."""
    prefixes = {
        'preprocessing.filters': 'pre.hpf.weight',
        'block_1a.': 'group1.basic.',
        'block_2a.': 'group2.',
        'block_3.': 'group3.',
        'block_2b.': 'group4.',
        'block_1b.': 'group5.basic.',
        'fc.': 'fc1.',
    }
    converted = {}
    for k, v in state_dict.items():
        old = next(old for old in prefixes if k.startswith(old))
        if old == 'preprocessing.filters':
            v = v[:62]  # reference applies the same filters to each channel
        converted[prefixes[old] + k[len(old):]] = v
    return converted


class TestUcNet(unittest.TestCase):
    """Test suite for ucnet module."""
    _logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        try:
            import cv2  # noqa: F401
            cls.reference = load_reference(cls._tmpdir.name)
        except (ImportError, OSError) as e:
            cls.reference = None
            cls._reference_error = str(e)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _require_reference(self):
        if self.reference is None:
            self.skipTest(f'reference not available: {self._reference_error}')

    def test_structure(self):
        """Checks the shapes against the Tab. 2 in the paper."""
        self._logger.info('TestUcNet.test_structure()')
        model = sw.ucnet.UcNet()
        #
        x = torch.ones((1, 3, 256, 256), dtype=torch.float32)
        x_preprocessing = model.preprocessing(x)
        self.assertEqual(tuple(x_preprocessing.size()), (1, 186, 256, 256))
        #
        x_1a = model.block_1a(x_preprocessing)
        self.assertEqual(tuple(x_1a.size()), (1, 32, 256, 256))
        #
        x_2a = model.block_2a(x_1a)
        self.assertEqual(tuple(x_2a.size()), (1, 32, 128, 128))
        #
        x_3 = model.block_3(x_2a)
        self.assertEqual(tuple(x_3.size()), (1, 64, 64, 64))
        #
        x_2b = model.block_2b(x_3)
        self.assertEqual(tuple(x_2b.size()), (1, 128, 32, 32))
        #
        x_1b = model.block_1b(x_2b)
        self.assertEqual(tuple(x_1b.size()), (1, 256, 32, 32))
        #
        x_gap = model.gap(x_1b)
        self.assertEqual(tuple(x_gap.size()), (1, 256))
        x_fc = model.fc(x_gap)
        self.assertEqual(tuple(x_fc.size()), (1, 2))

    def test_filters_match_reference(self):
        self._logger.info('TestUcNet.test_filters_match_reference()')
        self._require_reference()
        np.testing.assert_array_equal(
            np.array(self.reference.build_filters()),
            sw.ucnet._ucnet.build_filters(),
        )

    @parameterized.expand([['random'], ['pretrained']])
    def test_compatible_with_reference(self, weights: str):
        """Compares against implementation by the authors."""
        self._logger.info(f'TestUcNet.test_compatible_with_reference({weights=})')
        self._require_reference()
        torch.manual_seed(12345)
        if weights == 'pretrained':
            model = sw.ucnet.UcNet.pretrained()
        else:
            model = sw.ucnet.UcNet()
            # non-trivial BN statistics
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.uniform_(-1, 1)
                    m.running_var.uniform_(.5, 2)
        model_ref = self.reference.Net()
        model_ref.load_state_dict(to_reference(model.state_dict()), strict=True)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model_ref.parameters() if p.requires_grad),
        )
        #
        x = torch.rand((2, 3, 256, 256), dtype=torch.float32)
        for train in [False, True]:
            model.train(train)
            model_ref.train(train)
            with torch.no_grad():
                torch.testing.assert_close(model(x), model_ref(x.clone()))

    def test_pretrained(self):
        self._logger.info('TestUcNet.test_pretrained()')
        model = sw.ucnet.UcNet.pretrained()
        self.assertFalse(model.training)

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_infere_ucnet(self, fname: str):
        """Smoke test, weights are trained for J-UNIWARD, not for LSB matching."""
        self._logger.info(f'TestUcNet.test_infere_ucnet({fname=})')
        #
        DEVICE = torch.device('cpu')
        model = sw.ucnet.UcNet.pretrained(device=DEVICE)

        #
        cover_path = defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'
        x0 = np.array(Image.open(cover_path))
        y0 = sw.ucnet.infere_single(x0, model=model)
        #
        x1 = cl.lsb.simulate(x0, alpha=1., modify=cl.LSB_MATCHING, seed=12345)
        y1 = sw.ucnet.infere_single(x1, model=model)
        for y in [y0, y1]:
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 1)
