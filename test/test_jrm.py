import unittest
from parameterized import parameterized
from sealwatch.features.jrm.jrm import extract_jrm_features_from_filepath
from scipy.io import loadmat
import numpy as np
import re
import os

import defs

FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'jrm'

# BASE_DIR = "assets"


class TestJRM(unittest.TestCase):

    # Map group names in Matlab to group prefixes in our implementation
    replace_submodel_names_map = {
        "D(?P<direction>[a-z]+)1_T(?P<T>[0-9])": "intra_block_hv_D\g<direction>_T\g<T>", # Replace "Dh1_T2" with "intra_block_hv_Dh_T2"
        "D(?P<direction>[a-z]+)2_T(?P<T>[0-9])": "intra_block_diag_D\g<direction>_T\g<T>",  # Replace "Dh2_T2" with "intra_block_diag_Dh_T2"
        "D(?P<direction>[a-z]+)3_T(?P<T>[0-9])": "inter_block_hv_D\g<direction>_T\g<T>", # Replace "Dh3_T2" with "inter_block_hv_Dh_T2"
    }

    # @parameterized.expand([
    #     ("cover/jpeg_75_gray/seal1.jpg", "features_matlab/jrm/seal1.mat"),
    #     ("cover/jpeg_75_gray/seal2.jpg", "features_matlab/jrm/seal2.mat"),
    #     ("cover/jpeg_75_gray/seal3.jpg", "features_matlab/jrm/seal3.mat"),
    #     ("cover/jpeg_75_gray/seal4.jpg", "features_matlab/jrm/seal4.mat"),
    #     ("cover/jpeg_75_gray/seal5.jpg", "features_matlab/jrm/seal5.mat"),
    #     ("cover/jpeg_75_gray/seal6.jpg", "features_matlab/jrm/seal6.mat"),
    #     ("cover/jpeg_75_gray/seal7.jpg", "features_matlab/jrm/seal7.mat"),
    #     ("cover/jpeg_75_gray/seal8.jpg", "features_matlab/jrm/seal8.mat"),
    #     ("cover/jpeg_75_gray/otter1.jpg", "features_matlab/jrm/otter1.mat"),
    #     ("cover/jpeg_75_gray/otter2.jpg", "features_matlab/jrm/otter2.mat"),
    #     ("cover/jpeg_75_gray/dolphin.jpg", "features_matlab/jrm/dolphin.mat"),
    # ])
    # def test_compare_matlab(self, cover_filepath, matlab_features_filepath):
    #     npy_jrm_features = extract_jrm_features_from_filepath(os.path.join(BASE_DIR, cover_filepath))
    #     matlab_jrm_features = loadmat(os.path.join(BASE_DIR, matlab_features_filepath))

    #     matlab_submodel_names = matlab_jrm_features["features"].dtype.names
    #     for matlab_submodel_name in matlab_submodel_names:
    #         # Skip features from Cartesian calibration, because the calibration is implemented slightly differently
    #         if matlab_submodel_name.endswith("_ref"):
    #             continue

    #         # Obtain Matlab submodel features
    #         matlab_submodel_features = matlab_jrm_features["features"][matlab_submodel_name][0][0].flatten()

    #         # Check whether we need to translate the submodel name
    #         npy_submodel_prefix = matlab_submodel_name
    #         for pattern, replacement in TestJRM.replace_submodel_names_map.items():
    #             # Translate submodel name
    #             if re.search(pattern, matlab_submodel_name) is not None:
    #                 npy_submodel_prefix = re.sub(pattern, replacement, matlab_submodel_name)
    #                 break

    #         # Find all submodels that start with the same prefix
    #         npy_submodel_features = []
    #         for key, value in npy_jrm_features.items():
    #             # Skip Cartesian calibration features
    #             if key.endswith("_ref"):
    #                 continue

    #             # Matching prefix: Extract features
    #             if key.startswith(npy_submodel_prefix):
    #                 npy_submodel_features.append(value)

    #         # No matching features
    #         if len(npy_submodel_features) == 0:
    #             raise ValueError(f"No matching features for Matlab submodel {matlab_submodel_name}")

    #         # Concatenate submodel's subfeatures to one feature array
    #         npy_submodel_features = np.concatenate(npy_submodel_features)

    #         # Compare to Matlab features
    #         np.testing.assert_allclose(npy_submodel_features, matlab_submodel_features)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_matlab(self, fname):
        npy_jrm_features = extract_jrm_features_from_filepath(defs.COVER_CG_DIR / f'{fname}.jpg')
        matlab_jrm_features = loadmat(FEATURES_DIR / f'{fname}.mat')

        matlab_submodel_names = matlab_jrm_features["features"].dtype.names
        for matlab_submodel_name in matlab_submodel_names:
            # Skip features from Cartesian calibration, because the calibration is implemented slightly differently
            if matlab_submodel_name.endswith("_ref"):
                continue

            # Obtain Matlab submodel features
            matlab_submodel_features = matlab_jrm_features["features"][matlab_submodel_name][0][0].flatten()

            # Check whether we need to translate the submodel name
            npy_submodel_prefix = matlab_submodel_name
            for pattern, replacement in TestJRM.replace_submodel_names_map.items():
                # Translate submodel name
                if re.search(pattern, matlab_submodel_name) is not None:
                    npy_submodel_prefix = re.sub(pattern, replacement, matlab_submodel_name)
                    break

            # Find all submodels that start with the same prefix
            npy_submodel_features = []
            for key, value in npy_jrm_features.items():
                # Skip Cartesian calibration features
                if key.endswith("_ref"):
                    continue

                # Matching prefix: Extract features
                if key.startswith(npy_submodel_prefix):
                    npy_submodel_features.append(value)

            # No matching features
            if len(npy_submodel_features) == 0:
                raise ValueError(f"No matching features for Matlab submodel {matlab_submodel_name}")

            # Concatenate submodel's subfeatures to one feature array
            npy_submodel_features = np.concatenate(npy_submodel_features)

            # Compare to Matlab features
            np.testing.assert_allclose(npy_submodel_features, matlab_submodel_features)


__all__ = ["TestJRM"]
