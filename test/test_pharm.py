"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from itertools import product
import numpy as np
from parameterized import parameterized
import scipy.io
import sealwatch as sw
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'pharm'


def load_test_cases():
    proj_mat_heights = np.arange(1, 4)
    proj_mat_widths = np.arange(1, 4)
    shift_ys = np.arange(8)
    shift_xs = np.arange(8)

    test_cases = product(proj_mat_heights, proj_mat_widths, shift_ys, shift_xs)

    return list(test_cases)


class TestPharm(unittest.TestCase):

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname, num_projections=900):
        """Default (batched) against the Matlab reference."""
        f = sw.pharm.extract_from_file(
            defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg',
            num_projections=num_projections,
            implementation=sw.PHARM_ORIGINAL,
        )
        f_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')

        for model in f_ref["features"].dtype.names:
            np.testing.assert_array_equal(
                f[model].flatten(),
                f_ref["features"][model][0][0].flatten(),
            )

    @parameterized.expand([
        (impl, image, kw)
        for impl in ['original', 'fix']
        for image, kw in [
            ('seal1', {}),
            ('seal1', {'symmetrize': False}),
            ('seal1', {'first_order_residuals': False}),
            ('seal1', {'second_order_residuals': False, 'third_order_residuals': False}),
            ('seal1', {'T': 3, 'q': 2.5, 'seed': 7, 'maximum_projection_size': 3}),
            ('nonsquare', {}),
            ('odd', {}),
        ]
    ])
    def test_former_extractors(self, impl, image, kw):
        """Unified implementation, not batched, is bit-identical to the former extractor classes."""
        implementation = {'original': sw.PHARM_ORIGINAL, 'fix': sw.PHARM_FIX}[impl]
        rng = np.random.default_rng(12345)
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg')
        x = {
            'seal1': x,
            'nonsquare': x[:300, ::2],
            'odd': rng.normal(128, 30, (61, 45)),
        }[image]
        f = sw.pharm.extract(x, implementation=implementation, num_projections=20, batched=False, **kw)
        f_ref = sw.pharm.get_extractor_factory(implementation)(num_projections=20, **kw).extract(x)
        self.assertEqual(list(f), list(f_ref))
        for k in f:
            self.assertEqual(f[k].dtype, f_ref[k].dtype, k)
            np.testing.assert_array_equal(f[k], f_ref[k], err_msg=k)

    @parameterized.expand([
        (impl, image, kw)
        for impl in ['original', 'fix']
        for image, kw in [
            ('seal1', {}),
            ('seal1', {'symmetrize': False, 'normalize': True}),
            ('seal1', {'T': 3, 'q': 2.5, 'seed': 7, 'maximum_projection_size': 3}),
            ('nonsquare', {}),
            ('odd', {}),
        ]
    ])
    def test_batched(self, impl, image, kw):
        """Batched projections give the same features, on these images."""
        implementation = {'original': sw.PHARM_ORIGINAL, 'fix': sw.PHARM_FIX}[impl]
        rng = np.random.default_rng(12345)
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg')
        x = {
            'seal1': x,
            'nonsquare': x[:300, ::2],
            'odd': rng.normal(128, 30, (61, 45)),
        }[image]
        f = sw.pharm.extract(x, implementation=implementation, num_projections=300, batched=False, **kw)
        f_batched = sw.pharm.extract(x, implementation=implementation, num_projections=300, **kw)
        self.assertEqual(list(f), list(f_batched))
        for k in f:
            self.assertEqual(f[k].dtype, f_batched[k].dtype, k)
            np.testing.assert_array_equal(f[k], f_batched[k], err_msg=k)

    @parameterized.expand([['seal1'], ['nonsquare'], ['odd']])
    def test_batched_precision(self, image):
        """Batched convolutions equal the strided convolutions, up to rounding."""
        rng = np.random.default_rng(12345)
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg')
        x = {
            'seal1': x,
            'nonsquare': x[:300, ::2],
            'odd': rng.normal(128, 30, (61, 45)),
        }[image]
        # 3rd order residual, of the largest values
        residual = x[:, :-3] - 3 * x[:, 1:-2] + 3 * x[:, 2:-1] - x[:, 3:]
        # all kernel sizes and offsets
        kernels = [
            (rng.normal(size=(h, w)), offset_y, offset_x)
            for h in range(1, 9) for w in range(1, 9) for offset_y in range(8) for offset_x in range(8)
        ]
        proj_batched = []
        for proj, valid in sw.pharm._pharm._strided_convolutions_batched(residual, kernels):
            for i in range(proj.shape[-1]):
                rows, cols = np.nonzero(valid[..., i].any(axis=1))[0], np.nonzero(valid[..., i].any(axis=0))[0]
                proj_batched.append(proj[:len(rows), :len(cols), i])
        self.assertEqual(len(proj_batched), len(kernels))
        for (kernel, offset_y, offset_x), proj in zip(kernels, proj_batched):
            proj_ref = sw.tools.signal.strided_convolution(residual, kernel, stride=8, offset_y=offset_y, offset_x=offset_x)
            self.assertEqual(proj.shape, proj_ref.shape)
            # outputs up to about 1e3, double precision
            np.testing.assert_allclose(proj, proj_ref, rtol=0, atol=1e-10)

    def test_compare_matlab_unbatched(self):
        """Projections one at a time against the Matlab reference, on a single image."""
        f = sw.pharm.extract_from_file(
            defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg',
            num_projections=900,
            implementation=sw.PHARM_ORIGINAL,
            batched=False,
        )
        f_ref = scipy.io.loadmat(FEATURES_DIR / 'seal1.mat')
        for model in f_ref["features"].dtype.names:
            np.testing.assert_array_equal(f[model].flatten(), f_ref["features"][model][0][0].flatten())

    def test_legacy_name(self):
        self.assertIs(sw.PHARM_REVISITED, sw.PHARM_FIX)
        self.assertIs(sw.pharm.Implementation.PHARM_REVISITED, sw.pharm.Implementation.PHARM_FIX)

    def test_normalize(self):
        x = sw.tools.jpeg.decompress_from_file(defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg')
        for implementation in [sw.PHARM_ORIGINAL, sw.PHARM_FIX]:
            for symmetrize in [False, True]:
                f = sw.pharm.extract(x, implementation=implementation, num_projections=5, symmetrize=symmetrize)
                f_norm = sw.pharm.extract(x, implementation=implementation, num_projections=5, symmetrize=symmetrize, normalize=True)
                for k in f:
                    self.assertEqual(f_norm[k].dtype, np.float64)
                    # each histogram is divided by the number of projected blocks, about (512 / 8)^2
                    np.testing.assert_allclose(f_norm[k] * 64 ** 2, f[k], rtol=.05)

    def test_qf(self):
        path = defs.COVER_COMPRESSED_GRAY_DIR / 'seal1.jpg'
        self.assertEqual(sw.pharm.qf_to_quantization_step(75), 5)
        f_qf = sw.pharm.extract_from_file(path, qf=75, num_projections=5)
        f_q = sw.pharm.extract_from_file(path, q=5, num_projections=5)
        f_default = sw.pharm.extract_from_file(path, num_projections=5)
        f_95 = sw.pharm.extract_from_file(path, qf=95, num_projections=5)
        for k in f_qf:
            np.testing.assert_array_equal(f_qf[k], f_q[k])
            np.testing.assert_array_equal(f_qf[k], f_default[k])
        self.assertFalse(all(np.array_equal(f_qf[k], f_95[k]) for k in f_qf))
        with self.assertRaises(TypeError):
            sw.pharm.extract_from_file(path, qf=75, q=5)

    def test_deprecated_x1(self):
        x = np.random.default_rng(12345).normal(128, 30, (40, 40))
        with self.assertWarns(DeprecationWarning):
            f_x1 = sw.pharm.extract(x1=x, num_projections=5)
        f_x = sw.pharm.extract(x=x, num_projections=5)
        for k in f_x:
            np.testing.assert_array_equal(f_x1[k], f_x[k])

    @staticmethod
    def _compare_histograms(residual, kernel_height, kernel_width, shift_y, shift_x, proj_mat):
        extractor = sw.pharm.get_extractor_factory(sw.PHARM_FIX)(num_projections=10, T=3, q=1)
        h_original, h_vertical_flip, h_horizontal_flip, h_rot180 = extractor._obtain_histograms_to_merge(
            residual=residual,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat,
        )

        assert np.all(h_original == h_vertical_flip)
        assert np.all(h_original == h_horizontal_flip)
        assert np.all(h_original == h_rot180)

    def _compare_first_order_kernels(self, img, shift_y, shift_x, proj_mat):
        # Right: [-1, 1] in horizontal direction
        R = -img[:, :-1] + img[:, 1:]

        self._compare_histograms(
            residual=R,
            kernel_height=1,
            kernel_width=2,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

        # Down: [-1, 1] in vertical direction
        D = -img[:-1, :] + img[1:, :]

        self._compare_histograms(
            residual=D,
            kernel_height=2,
            kernel_width=1,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

    def _compare_second_order_kernels(self, img, shift_y, shift_x, proj_mat):
        # (2.1) Vertical direction:
        # [[ 1,  1],
        #  [-1, -1]]
        Dh = (img[:-1, :-1]  # left top
              + img[:-1, 1:]  # right top
              - img[1:, :-1]  # left bottom
              - img[1:, 1:])  # right bottom

        self._compare_histograms(
            residual=Dh,
            kernel_height=2,
            kernel_width=2,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

        # (2.2) Horizontal direction
        # [[-1,  1],
        #  [-1,  1]]
        Dv = (- img[:-1, :-1]  # left top
              + img[:-1, 1:]  # right top
              - img[1:, :-1]  # left bottom
              + img[1:, 1:])  # right bottom

        self._compare_histograms(
            residual=Dv,
            kernel_height=2,
            kernel_width=2,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

        # (2.3) Diagonal direction
        # [[ 1, -1],
        #  [-1,  1]]
        Dd = (+ img[:-1, :-1]  # left top
              - img[:-1, 1:]  # right stop
              - img[1:, :-1]  # left bottom
              + img[1:, 1:])  # right bottom

        self._compare_histograms(
            residual=Dd,
            kernel_height=2,
            kernel_width=2,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

    def _compare_third_order_kernels(self, img, shift_y, shift_x, proj_mat):
        # Right: [1, -3, 3, -1]
        R = + img[:, :-3] - 3 * img[:, 1:-2] + 3 * img[:, 2:-1] - img[:, 3:]

        self._compare_histograms(
            residual=R,
            kernel_height=1,
            kernel_width=4,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

        # Down: [1, -3, 3, -1]
        D = + img[:-3, :] - 3 * img[1:-2] + 3 * img[2:-1, :] - img[3:, :]

        self._compare_histograms(
            residual=D,
            kernel_height=4,
            kernel_width=1,
            shift_y=shift_y,
            shift_x=shift_x,
            proj_mat=proj_mat
        )

    @parameterized.expand(
        # Each tuple contains (proj_mat_height, proj_mat_width, shift_y, shift_x)
        load_test_cases()
    )
    def test_symmetric_img(self, proj_mat_height, proj_mat_width, shift_y, shift_x):
        rng = np.random.default_rng(12345)

        # Create symmetric blocks
        block = rng.random(size=(8, 8))
        block = (block + np.transpose(block, axes=(1, 0))) / 2
        block = (block + block[::-1, :]) / 2
        block = (block + block[:, ::-1]) / 2

        assert np.allclose(block, np.transpose(block, axes=(1, 0)))
        assert np.allclose(block, block[::-1, :])
        assert np.allclose(block, block[:, ::-1])

        img = np.tile(block, reps=[8, 8])

        # Generate random projection matrix
        proj_mat_shape = (proj_mat_height, proj_mat_width)
        proj_mat = rng.normal(size=proj_mat_shape)
        # Normalize so that the Frobenius norm of the projection matrix is 1
        proj_mat = proj_mat / np.sqrt(np.sum(proj_mat ** 2))

        self._compare_first_order_kernels(img=img, shift_y=shift_y, shift_x=shift_x, proj_mat=proj_mat)

        self._compare_second_order_kernels(img=img, shift_y=shift_y, shift_x=shift_x, proj_mat=proj_mat)

        self._compare_third_order_kernels(img=img, shift_y=shift_y, shift_x=shift_x, proj_mat=proj_mat)


__all__ = ["TestPharm"]
