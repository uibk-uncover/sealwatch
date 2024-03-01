import unittest
from sealwatch.features.pharm.pharm_original import extract_pharm_original_features
from sealwatch.features.pharm.pharm_revisited import PharmRevisitedFeatureExtractor, extract_pharm_revisited_features
from sealwatch.utils.grouping import flatten_single
from parameterized import parameterized
from scipy.io import loadmat
from itertools import product
import numpy as np
import os


BASE_DIR = "tests/assets"


def load_test_cases():
    proj_mat_heights = np.arange(1, 4)
    proj_mat_widths = np.arange(1, 4)
    shift_ys = np.arange(8)
    shift_xs = np.arange(8)

    test_cases = product(proj_mat_heights, proj_mat_widths, shift_ys, shift_xs)

    return list(test_cases)


class TestPharm(unittest.TestCase):

    @parameterized.expand([
        ("cover/00001.jpeg", "features_matlab/00001_pharm_num_projections_20.mat", 20),
        ("cover/00002.jpeg", "features_matlab/00002_pharm_num_projections_20.mat", 20),
        ("cover/00003.jpeg", "features_matlab/00003_pharm_num_projections_20.mat", 20),
        ("cover/00004.jpeg", "features_matlab/00004_pharm_num_projections_20.mat", 20),
        ("cover/00005.jpeg", "features_matlab/00005_pharm_num_projections_20.mat", 20),
        # ("cover/00001.jpeg", "features_matlab/00001_pharm_num_projections_900.mat", 900), # these tests take very long
        # ("cover/00002.jpeg", "features_matlab/00002_pharm_num_projections_900.mat", 900),
        # ("cover/00003.jpeg", "features_matlab/00003_pharm_num_projections_900.mat", 900),
        # ("cover/00004.jpeg", "features_matlab/00004_pharm_num_projections_900.mat", 900),
        # ("cover/00005.jpeg", "features_matlab/00005_pharm_num_projections_900.mat", 900),
    ])
    def test_compare_matlab(self, cover_filepath, matlab_features_filepath, num_projections):
        pharm_features = extract_pharm_original_features(os.path.join(BASE_DIR, cover_filepath), quantization_step=5, T=2, num_projections=num_projections)
        matlab_pharm_features = loadmat(os.path.join(BASE_DIR, matlab_features_filepath))

        matlab_submodel_names = matlab_pharm_features["f"].dtype.names
        for matlab_submodel_name in matlab_submodel_names:
            # Obtain Matlab submodel features
            matlab_submodel_features = matlab_pharm_features["f"][matlab_submodel_name][0][0].flatten()

            # Obtain Python submodel features
            pharm_submodel_features = pharm_features[matlab_submodel_name].flatten()

            # Compare
            np.testing.assert_allclose(pharm_submodel_features, matlab_submodel_features)

    @staticmethod
    def _compare_histograms(residual, kernel_height, kernel_width, shift_y, shift_x, proj_mat):
        extractor = PharmRevisitedFeatureExtractor(num_projections=10, T=3, quantization_step=1)
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

    @parameterized.expand([
        ("cover/00001.jpeg",),
        ("cover/00002.jpeg",),
        ("cover/00003.jpeg",),
        ("cover/00004.jpeg",),
        ("cover/00005.jpeg",),
    ])
    def test_compare_original_and_revisited_no_symmetrize(self, img_filepath):
        feature_args = {
            "img_filepath": os.path.join(BASE_DIR, img_filepath),
            "num_projections": 100,
            "quantization_step": 4,
            "T": 3,
            "symmetrize": False,
        }

        original_features = extract_pharm_original_features(**feature_args)
        revisited_features = extract_pharm_revisited_features(**feature_args)

        original_features = flatten_single(original_features)
        revisited_features = flatten_single(revisited_features)

        np.testing.assert_allclose(original_features, revisited_features)


__all__ = ["TestPharm"]
