"""
Implementation of the PHARM features as described in

Vojtěch Holub and Jessica Fridrich
"Phase-aware projection model for steganalysis of JPEG images"
IS&T/SPIE Electronic Imaging, 2015
https://doi.org/10.1117/12.2075239

Author: Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2015 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501


from collections import OrderedDict
import numpy as np
from pathlib import Path
from typing import Union

from .. import tools


class Extractor(object):
    def __init__(
        self,
        q: int = 5,
        T: int = 2,
        num_projections: int = 100,
        maximum_projection_size: int = 8,
        first_order_residuals: bool = True,
        second_order_residuals: bool = True,
        third_order_residuals: bool = True,
        symmetrize: bool = True,
        normalize: bool = False,
        seed: int = 1
    ):
        """Implementation of the PHARM features described in [1].
        This implementation matches the original Matlab implementation.

        [1] V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, San Francisco, CA, February 8–12, 2015.
        http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

        :param q: quantization step
        :param T: histogram truncation threshold
        :param num_projections: number of random projection matrices. The original code defaults to 900, but we reduce the default to 100 for speed reasons.
        :param maximum_projection_size: maximum spatial size of each projection matrix
        :param first_order_residuals: If True, include first order residuals. If False, skip first order residuals.
        :param second_order_residuals: If True, include second order residuals. If False, skip second order residuals.
        :param third_order_residuals: If True, include third order residuals. If False, skip third order residuals.
        :param symmetrize: If True, merge histograms with horizontally and vertically flipped versions of the image. If False, skip symmetrization.
        :param normalize: If True, normalize the histogram counts.
        :param seed: seed for random number generator for the projection matrices
        """

        self.num_projections = num_projections
        self.q = q
        self.T = T
        self.maximum_projection_size = maximum_projection_size
        self.first_order_residuals = first_order_residuals
        self.second_order_residuals = second_order_residuals
        self.third_order_residuals = third_order_residuals
        self.symmetrize = symmetrize
        self.normalize = normalize
        self.seed = seed

    def extract_from_file(self, path: Union[str, Path]) -> OrderedDict:
        """
        Extract phase-aware projection rich model (PHARM) features from a given JPEG image file.
        :param path: image flepath
        :return: ordered dict with individual submodels
        """
        # Decompress image without rounding
        img = tools.jpeg.decompress_from_file(path)
        return self.extract(img)

    def extract(self, img: np.ndarray) -> OrderedDict:
        """
        Extract phase-aware projection rich model (PHARM) features.

        The total number of features is: 7 filter kernels * T histogram values * num_random_projections = 7 * 2 * 900 = 12600

        :param img: decompressed JPEG image
        :return: ordered dict with individual submodels. The keys are the submodel names and the values are the features of shape [num_projections, T]. Note that the submodels are not normalized.
        """
        raise NotImplementedError


class OriginalExtractor(Extractor):
    """Implementation of PHARM features matching the original Matlab."""

    def _obtain_histograms_to_merge(self, residual, kernel_size, shift_y, shift_x, proj_mat):
        """
        Calculate the four histograms at the given offset (shift_y, shift_x) and at the horizontally and vertically flipped locations
        Assumes that residual is aligned with top-left corner of the original image.
        :param residual: noise residual; expected to be aligned with the top-left corner of the original image
        :param kernel_size: kernel size
        :param shift_y: vertical offset of the projection matrix
        :param shift_x: horizontal offset of the projection matrix
        :param proj_mat: projection matrix
        :return: histograms to be merged as 4-tuple (h_original, h_vertical_flip, h_horizontal_flip, h_rot180). Each histogram contains T entries.
        """

        proj_mat_height, proj_mat_width = proj_mat.shape

        # Set up bin edges from [0, T]
        bin_edges = np.arange(0, self.T + 1)
        # Shift bin edges to [0, 1 * q, 2 * q, ..., T * q]
        bin_edges = bin_edges * self.q

        # (1) Original orientation
        proj = tools.signal.strided_convolution(
            img=residual,
            kernel=proj_mat,
            stride=8,
            offset_y=shift_y,
            offset_x=shift_x
        )
        # Compute the histogram
        h_original = np.histogram(np.abs(proj.flatten()), bin_edges)[0]
        if self.normalize:
            # This will auto-cast to float
            h_original /= proj.size

        if not self.symmetrize:
            # No symmetrization? Stop here
            return h_original

        # (2) Vertical kernel flip
        # Shift is one lower than in Matlab
        offset_y = -1 - shift_y - proj_mat_height + kernel_size
        offset_x = shift_x

        while offset_y < 0:
            offset_y += 8

        proj = tools.signal.strided_convolution(
            img=residual,
            kernel=np.flipud(proj_mat),
            stride=8,
            offset_y=offset_y,
            offset_x=offset_x)

        h_vertical_flip = np.histogram(np.abs(proj.flatten()), bin_edges)[0]
        if self.normalize:
            h_vertical_flip /= proj.size

        # (3) Horizontal kernel flip
        offset_y = shift_y
        offset_x = -1 - shift_x - proj_mat_width + kernel_size

        while offset_x < 0:
            offset_x += 8

        proj = tools.signal.strided_convolution(
            img=residual,
            kernel=np.fliplr(proj_mat),
            stride=8,
            offset_y=offset_y,
            offset_x=offset_x,
        )

        h_horizontal_flip = np.histogram(np.abs(proj.flatten()), bin_edges)[0]
        if self.normalize:
            h_horizontal_flip /= proj.size

        # (4) Both horizontal and vertical kernel flip
        offset_y = -1 - shift_y - proj_mat_height + kernel_size
        offset_x = -1 - shift_x - proj_mat_width + kernel_size

        while offset_y < 0:
            offset_y += 8
        while offset_x < 0:
            offset_x += 8

        proj = tools.signal.strided_convolution(
            img=residual,
            kernel=np.flipud(np.fliplr(proj_mat)),
            stride=8,
            offset_y=offset_y,
            offset_x=offset_x
        )

        h_rot180 = np.histogram(np.abs(proj.flatten()), bin_edges)[0]
        if self.normalize:
            h_rot180 /= proj.size

        return h_original, h_vertical_flip, h_horizontal_flip, h_rot180

    def _proj_hist_spam(self, residual, kernel_size, rng):
        """
        Generate a number of random projections and apply the projections
        :param residual: 2D residual
        :param kernel_size: kernel size?
        :param rng: random number generator
        :return: features of shape [num_projections, T]
        """

        output_dtype = int
        if self.normalize:
            output_dtype = float

        # Allocate space for output features
        h = np.zeros((self.num_projections, self.T), dtype=output_dtype)

        for proj_idx in range(self.num_projections):
            # Randomly select height and width of projection matrix, each between [1, s]
            proj_mat_shape = tools.matlab.randi(low=1, high=self.maximum_projection_size + 1, size=2, rng=rng)

            # Randomly select grid offset
            shift_y, shift_x = tools.matlab.randi(low=0, high=8, size=2, rng=rng)

            # Generate random projection matrix. Reshape to match Matlab order
            proj_mat = tools.matlab.randn(proj_mat_shape, rng=rng).flatten(order="C").reshape(proj_mat_shape, order="F")

            # Normalize so that the Frobenius norm of the projection matrix is 1
            proj_mat = proj_mat / np.sqrt(np.sum(proj_mat ** 2))

            # Obtain histograms: If self.symmetrize, histograms will be a 4-tuple. Otherwise, histograms will be a single histogram.
            histograms = self._obtain_histograms_to_merge(
                residual=residual,
                kernel_size=kernel_size,
                shift_y=shift_y,
                shift_x=shift_x,
                proj_mat=proj_mat,
            )

            if self.symmetrize:
                h_original, h_vertical_flip, h_horizontal_flip, h_rot180 = histograms
                histograms = h_original + h_vertical_flip + h_horizontal_flip + h_rot180

            h[proj_idx, :] = histograms

        return h

    def extract(self, img: np.ndarray) -> OrderedDict:
        next_seed = self.seed

        features = OrderedDict()

        # First order kernels
        if self.first_order_residuals:
            # [-1, 1] in horizontal and vertical directions
            features.update(self._all_first(img, next_seed))
        next_seed += 2

        # Third order kernels
        if self.third_order_residuals:
            # [1, -3, 3, -1] in horizontal and vertical directions
            features.update(self._all_third(img, next_seed))
        next_seed += 2

        # Second order kernels
        if self.second_order_residuals:
            # (2.1) Vertical direction:
            # [[ 1,  1],
            #  [-1, -1]]

            # (2.2) Horizontal direction
            # [[-1,  1],
            #  [-1,  1]]

            # (2.3) Diagonal direction
            # [[ 1, -1],
            #  [-1,  1]]
            features.update(self._all_2x2(img, next_seed))
        next_seed += 3

        return features

    def _all_first(self, img, seed):
        kernel_size = 3
        height, width = img.shape

        # Cut off one pixel at all sides in order to avoid out-of-bounds accesses
        I = slice(1, height - 1)
        J = slice(1, width - 1)

        # Right: [-1, 1] in horizontal direction
        R = -img[I, J] + img[I, J.start + 1:J.stop + 1]

        # Up: [-1, 1] in vertical direction; the implementation swaps the sign
        U = img[I.start - 1:I.stop - 1, J] - img[I, J]

        features = OrderedDict()
        features["s1_spam14_R"] = self._proj_hist_spam(R, kernel_size, rng=np.random.RandomState(seed))
        features["s1_spam14_U"] = self._proj_hist_spam(U, kernel_size, rng=np.random.RandomState(seed + 1))
        return features

    def _all_third(self, img, seed):
        kernel_size = 5
        height, width = img.shape

        # Cut off two pixels at all sides in order to avoid out-of-bounds accesses
        I = slice(2, height - 2)
        J = slice(2, width - 2)

        # Right: [1, -3, 3, -1]
        R = + img[I, J.start - 1:J.stop - 1] - 3 * img[I, J] + 3 * img[I, J.start + 1:J.stop + 1] - img[I, J.start + 2:J.stop + 2]

        # Up: [1, -3, 3, -1]; the implementation swaps the sign
        U = -img[I.start - 2:I.stop - 2, J] + 3 * img[I.start - 1:I.stop - 1, J] - 3 * img[I, J] + img[I.start + 1:I.stop + 1, J]

        features = OrderedDict()
        features["s3_spam14_R"] = self._proj_hist_spam(R, kernel_size, rng=np.random.RandomState(seed))
        features["s3_spam14_U"] = self._proj_hist_spam(U, kernel_size, rng=np.random.RandomState(seed + 1))
        return features

    def _all_2x2(self, img, seed):
        kernel_size = 3
        height, width = img.shape

        # Cut off one pixel at all sides in order to avoid out-of-bounds accesses
        I = slice(1, height - 1)
        J = slice(1, width - 1)

        # (2.1) Vertical direction:
        # [[ 1,  1],
        #  [-1, -1]]
        # X(I-1,J-1) + X(I-1,J) - X(I,J-1) - X(I,J)
        Dh = (
            + img[I.start - 1:I.stop - 1, J.start - 1:J.stop - 1]  # left top
            + img[I.start - 1:I.stop - 1, J]  # right top
            - img[I, J.start - 1:J.stop - 1]  # left bottom
            - img[I, J])  # right bottom

        # (2.2) Horizontal direction; the implemention swaps the signs
        # [[-1,  1],
        #  [-1,  1]]
        # X(I-1,J-1) - X(I-1,J) + X(I,J-1) - X(I,J)
        Dv = (
            + img[I.start - 1:I.stop - 1, J.start - 1:J.stop - 1]  # left top
            - img[I.start - 1:I.stop - 1, J]  # right top
            + img[I, J.start - 1:J.stop - 1]  # left bottom
            - img[I, J])  # right bottom

        # (2.3) Diagonal direction; the implementation swaps the signs
        # [[ 1, -1],
        #  [-1,  1]]
        # -X(I-1,J-1)+X(I-1,J)+X(I,J-1)-X(I,J)
        Dd = (
            - img[I.start - 1:I.stop - 1, J.start - 1:J.stop - 1]  # left top
            + img[I.start - 1:I.stop - 1, J]  # right stop
            + img[I, J.start - 1:J.stop - 1]  # left bottom
            - img[I, J])  # right bottom

        features = OrderedDict()
        features["s2x2_spam14_H"] = self._proj_hist_spam(Dh, kernel_size, rng=np.random.RandomState(seed))
        features["s2x2_spam14_V"] = self._proj_hist_spam(Dv, kernel_size, rng=np.random.RandomState(seed + 1))
        features["s2x2_spam14_DMaj"] = self._proj_hist_spam(Dd, kernel_size, rng=np.random.RandomState(seed + 2))
        return features

    # @staticmethod
    # def qf_to_quantization_step(qf):
    #     quantization_step = (65 / 4) - (3 / 20) * qf
    #     return quantization_step

    # @staticmethod
    # def select_quantization_step(img_filepath):
    #     qf = tools.jpeg.identify_qf(img_filepath)
    #     return PharmOriginalFeatureExtractor.qf_to_quantization_step(qf)


# def extract_original(
#     img: np.ndarray,
#     q: int = 5,
#     T: int = 2,
#     num_projections: int = 100,
#     maximum_projection_size: int = 8,
#     first_order_residuals: bool = True,
#     second_order_residuals: bool = True,
#     third_order_residuals: bool = True,
#     symmetrize: bool = True,
#     normalize: bool = False,
# ):
#     """

#     Reference implementation of the PHARM features described in [1].
#     This implementation matches the Matlab implementation.

#     [1] V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, San Francisco, CA, February 8–12, 2015.
#     http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

#     :param img: decompressed JPEG image
#     :type img:
#     :param q: quantization step
#     :type q:
#     :param T: truncation threshold
#     :type T:
#     :param num_projections: number of random projection matrices. The original implementation defaults to 900, but we use 100 for speed reasons.
#     :type num_projections:
#     :param maximum_projection_size: maximum spatial size of each projection matrix
#     :type maximum_projection_size:
#     :param first_order_residuals: If True, include first order residuals. If False, skip first order residuals.
#     :type first_order_residuals:
#     :param second_order_residuals: If True, include second order residuals. If False, skip second order residuals.
#     :type second_order_residuals:
#     :param third_order_residuals: If True, include third order residuals. If False, skip third order residuals.
#     :type third_order_residuals:
#     :param symmetrize: If True, merge histograms with horizontally and vertically flipped versions of the image. If False, skip symmetrization.
#     :type symmetrize:
#     :param normalize: If True, normalize the histogram counts.
#     :type normalize:
#     :return: features as ordered dictionary, where the keys are the submodel names and the values are the features of shape [num_projections, T]. Note that the features are not normalized.
#     :rtype:
#     """
#     feature_extractor = PharmOriginalFeatureExtractor(
#         q=q,
#         T=T,
#         num_projections=num_projections,
#         maximum_projection_size=maximum_projection_size,
#         first_order_residuals=first_order_residuals,
#         second_order_residuals=second_order_residuals,
#         third_order_residuals=third_order_residuals,
#         symmetrize=symmetrize,
#         normalize=normalize,
#     )
#     return feature_extractor.extract(img)


# def extract_original_from_file(
#     path: Union[str, Path],
#     q=5,
#     T=2,
#     num_projections=100,
#     maximum_projection_size=8,
#     first_order_residuals=True,
#     second_order_residuals=True,
#     third_order_residuals=True,
#     symmetrize=True,
#     normalize=False,
# ):
#     """

#     Reference implementation of the PHARM features described in [1].
#     This implementation matches the Matlab implementation.

#     [1] V. Holub and J. Fridrich, Phase-Aware Projection Model for Steganalysis of JPEG Images, Proc. SPIE, Electronic Imaging, Media Watermarking, Security, and Forensics XVII, vol. 9409, San Francisco, CA, February 8–12, 2015.
#     http://dde.binghamton.edu/vholub/pdf/SPIE15_Phase-Aware_Projection_Model_for_Steganalysis_of_JPEG_Images.pdf

#     :param img_filepath: path to JPEG image
#     :param q: quantization step
#     :param T: truncation threshold
#     :param num_projections: number of random projection matrices. The original implementation defaults to 900, but we use 100 for speed reasons.
#     :param maximum_projection_size: maximum spatial size of each projection matrix
#     :param first_order_residuals: If True, include first order residuals. If False, skip first order residuals.
#     :param second_order_residuals: If True, include second order residuals. If False, skip second order residuals.
#     :param third_order_residuals: If True, include third order residuals. If False, skip third order residuals.
#     :param symmetrize: If True, merge histograms with horizontally and vertically flipped versions of the image. If False, skip symmetrization.
#     :param normalize: If True, normalize the histogram counts.
#     :return: features as ordered dictionary, where the keys are the submodel names and the values are the features of shape [num_projections, T]. Note that the features are not normalized.
#     """
#     feature_extractor = PharmOriginalFeatureExtractor(
#         q=q,
#         T=T,
#         num_projections=num_projections,
#         maximum_projection_size=maximum_projection_size,
#         first_order_residuals=first_order_residuals,
#         second_order_residuals=second_order_residuals,
#         third_order_residuals=third_order_residuals,
#         symmetrize=symmetrize,
#         normalize=normalize,
#     )
#     return feature_extractor.extract_from_file(path)
