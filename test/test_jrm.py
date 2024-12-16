
import jpeglib
import numpy as np
from parameterized import parameterized
import re
import scipy.io
import sealwatch as sw
import unittest

from . import defs


FEATURES_DIR = defs.ASSETS_DIR / 'features_matlab' / 'jrm'


class TestJRM(unittest.TestCase):

    # Map group names in Matlab to group prefixes in our implementation
    replace_submodel_names_map = {
        r'D(?P<direction>[a-z]+)1_T(?P<T>[0-9])': r'intra_block_hv_D\g<direction>_T\g<T>', # Replace "Dh1_T2" with "intra_block_hv_Dh_T2"
        r'D(?P<direction>[a-z]+)2_T(?P<T>[0-9])': r'intra_block_diag_D\g<direction>_T\g<T>',  # Replace "Dh2_T2" with "intra_block_diag_Dh_T2"
        r'D(?P<direction>[a-z]+)3_T(?P<T>[0-9])': r'inter_block_hv_D\g<direction>_T\g<T>', # Replace "Dh3_T2" with "inter_block_hv_Dh_T2"
    }

    def _test_extract(self, f, f_ref):

        for submodel in f_ref["features"].dtype.names:
            # Skip features from Cartesian calibration, because the calibration is implemented slightly differently
            if submodel.endswith("_ref"):
                continue

            # Check whether we need to translate the submodel name
            prefix = submodel
            for pattern, replacement in TestJRM.replace_submodel_names_map.items():
                # Translate submodel name
                if re.search(pattern, submodel) is not None:
                    prefix = re.sub(pattern, replacement, submodel)
                    break

            # Find all submodels that start with the same prefix
            f_jrm = []
            for key, value in f.items():
                # Skip Cartesian calibration features
                if key.endswith("_ref"):
                    continue

                # Matching prefix: Extract features
                if key.startswith(prefix):
                    f_jrm.append(value)

            # No matching features
            if len(f_jrm) == 0:
                raise ValueError(
                    f'No matching features for Matlab submodel {submodel}')

            # Concatenate submodel's subfeatures to one feature array
            f_jrm = np.concatenate(f_jrm)

            # Compare to Matlab features
            np.testing.assert_allclose(
                f_jrm,
                f_ref["features"][submodel][0][0].flatten(),
            )

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract_from_file(self, fname):
        #
        f = sw.jrm.extract_from_file(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        #
        f_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')
        #
        self._test_extract(f, f_ref)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        #
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        f = sw.jrm.extract(y1=jpeg0.Y, qt=jpeg0.qt[0])
        #
        f_ref = scipy.io.loadmat(FEATURES_DIR / f'{fname}.mat')
        #
        self._test_extract(f, f_ref)


__all__ = ["TestJRM"]
