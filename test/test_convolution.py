
import numpy as np
from parameterized import parameterized
import scipy.signal
# from scipy.signal import fftconvolve
import sealwatch as sw
# from sealwatch.utils.convolution import strided_convolution
import unittest


class TestConvolution(unittest.TestCase):

    @parameterized.expand(range(10))
    def test_strided_convolution(self, seed):
        stride = 8

        rng = np.random.default_rng(seed)
        img = rng.normal(size=(512, 512))

        rng = np.random.default_rng(seed)
        proj_mat_shape = rng.integers(low=1, high=8 + 1, size=2)
        proj_mat = rng.normal(size=proj_mat_shape)

        for offset_y in range(stride):
            for offset_x in range(stride):
                a = sw.tools.convolution.strided_convolution(img, proj_mat, stride=stride, offset_y=offset_y, offset_x=offset_x)
                b = scipy.signal.fftconvolve(img, proj_mat, mode="valid")[offset_y::stride, offset_x::stride]
                np.testing.assert_allclose(a, b)
