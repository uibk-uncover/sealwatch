import jpeglib


def decompress_crop_recompress(input_filepath, output_filepath):
    """
    Decompress the given image, crop 4 pixels from all sides, then recompress using the same quantization table
    :param input_filepath: path to JPEG image
    :param output_filepath: where to store the resulting JPEG images
    """

    # Decompress into spatial domain
    im_spatial = jpeglib.read_spatial(input_filepath)
    im_dct = jpeglib.read_dct(input_filepath)

    img = im_spatial.spatial

    # Crop 4 pixels from all sides
    img = img[4:-4, 4:-4]

    # Recompress
    im = jpeglib.from_spatial(img.copy())
    im.write_spatial(output_filepath, qt=im_dct.qt)
