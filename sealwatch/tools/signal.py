import numpy as np


def strided_correlation(img, kernel, stride, offset_y=0, offset_x=0):
    """
    Fast strided correlation
    :param img: 2D input array
    :param kernel: 2D kernel
    :param stride: stride of the correlation
    :param offset_y: optional offset in vertical direction
    :param offset_x: optional offset in horizontal direction
    :return: 2D array
    """

    if offset_y > 0:
        img = img[offset_y:]
    if offset_x > 0:
        img = img[:, offset_x:]

    kernel_height, kernel_width = kernel.shape
    input_height, input_width = img.shape

    output_height = 1 + (input_height - kernel_height) // stride
    output_width = 1 + (input_width - kernel_width) // stride

    output_shape = (output_height, output_width, kernel_height, kernel_width)
    strides = (img.strides[0] * stride, img.strides[1] * stride, img.strides[0], img.strides[1])
    input_view = np.lib.stride_tricks.as_strided(
        img,
        shape=output_shape,
        strides=strides
    )

    res = np.tensordot(input_view, kernel, axes=((2, 3), (0, 1)))

    return res


def strided_convolution(img, kernel, stride, offset_y=0, offset_x=0):
    """
    Fast strided convolution
    :param img: 2D input array
    :param kernel: 2D kernel
    :param stride: stride of the correlation
    :param offset_y: optional offset in vertical direction
    :param offset_x: optional offset in horizontal direction
    :return: 2D array
    """

    return strided_correlation(
        img=img,
        kernel=np.fliplr(np.flipud(kernel)),
        stride=stride,
        offset_y=offset_y,
        offset_x=offset_x
    )
