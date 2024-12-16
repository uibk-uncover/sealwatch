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

# from sealwatch.features.pharm import extract_pharm_original_features_from_file
# from sealwatch.features.pharm.pharm_revisited import PharmRevisitedFeatureExtractor

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
        f = sw.pharm.extract_from_file(
            defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg',
            num_projections=num_projections,
            implementation=sw.PHARM_ORIGINAL,
        )
        f_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')

        #
        for model in f_ref["features"].dtype.names:
            np.testing.assert_allclose(
                f[model].flatten(),
                f_ref["features"][model][0][0].flatten(),
            )

    @staticmethod
    def _compare_histograms(residual, kernel_height, kernel_width, shift_y, shift_x, proj_mat):
        extractor = sw.pharm.get_extractor_factory(sw.PHARM_REVISITED)(num_projections=10, T=3, q=1)
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
