import argparse
import jpeglib
from tqdm import tqdm
import numpy as np
from sealwatch.utils.logger import setup_custom_logger
from sealwatch.features.jrm.jrm import extract_cc_jrm_features_from_file, extract_jrm_features_from_file
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file
from sealwatch.features.pharm.pharm_original import extract_pharm_original_features_from_file
from sealwatch.features.pharm.pharm_revisited import extract_pharm_revisited_features_from_file
from sealwatch.features.spam.spam import extract_spam686_features_from_file
from sealwatch.features.dctr.dctr import extract_dctr_features_from_file
from sealwatch.utils.grouping import flatten_single
from sealwatch.utils.writer import BufferedWriter
from sealwatch.utils.quantization_table import identify_qf, create_qt_to_qf_mapping
from sealwatch.utils.constants import JRM, CC_JRM, GFR, PHARM_ORIGINAL, PHARM_REVISITED, SPAM, DCTR
from glob import glob
import os


log = setup_custom_logger(os.path.basename(__file__))


def extract_features(input_files, output_file, feature_type, feature_args):
    writer = BufferedWriter(output_filename=output_file, chunk_size=64)

    # Set up mapping from quantization table to quality factor
    qt_to_qf_map = None
    if feature_type in {GFR, DCTR}:
        # Peek into the first file to switch between color and grayscale
        im = jpeglib.read_dct(input_files[0])
        grayscale = im.num_components == 1

        if grayscale:
            log.info("Setting up QT-to-QF mapping for grayscale images")
        else:
            log.info("Setting up QF-to-QF mapping for color images")

        qt_to_qf_map = create_qt_to_qf_mapping(grayscale=grayscale)

    # Loop over input files
    for input_file in tqdm(input_files):

        try:
            if JRM == feature_type:
                features_grouped = extract_jrm_features_from_file(input_file)

                # Flatten grouped features
                features = flatten_single(features_grouped)

            elif CC_JRM == feature_type:
                features_grouped = extract_cc_jrm_features_from_file(input_file)

                # Flatten grouped features
                features = flatten_single(features_grouped)

            elif DCTR == feature_type:
                # Select quantization steps based on quality factor
                qf = identify_qf(input_file, qt_to_qf_map=qt_to_qf_map)
                if qf is None:
                    log.warning("Unknown JPEG quality. Setting quality factor to 75.")
                    qf = 75

                features = extract_dctr_features_from_file(input_file, qf=qf)

                # Flatten features
                features = features.flatten()

            elif GFR == feature_type:
                kwargs = {
                    "img_filepath": input_file
                }

                # Overwrite default parameters
                if feature_args.get("gfr_num_rotations"):
                    kwargs["num_rotations"] = feature_args["gfr_num_rotations"]

                if feature_args.get("gfr_truncation_threshold"):
                    kwargs["truncation_threshold"] = feature_args["gfr_truncation_threshold"]

                # Quantization steps or quality factor
                if feature_args.get("gfr_quantization_steps"):
                    assert len(feature_args["gfr_quantization_steps"]) == 4, "Expected four quantization steps"
                    kwargs["quantization_steps"] = feature_args["gfr_quantization_steps"]

                elif feature_args.get("gfr_quantization_step"):
                    # Only for the GFR_REVISITED_SINGLE_SCALE features
                    kwargs["quantization_step"] = feature_args["gfr_quantization_step"]

                else:
                    # Select quantization steps based on quality factor
                    # Identify QF from the input file
                    qf = identify_qf(input_file, qt_to_qf_map=qt_to_qf_map)
                    if qf is None:
                        log.warning("Unknown JPEG quality. Setting quality factor to 75.")
                        qf = 75

                    kwargs["qf"] = qf

                # Select feature variant
                features = extract_gfr_features_from_file(**kwargs)

                features = features.flatten()

            elif feature_type in {PHARM_ORIGINAL, PHARM_REVISITED}:
                kwargs = {
                    "img_filepath": input_file,
                    "first_order_residuals": True,
                    "second_order_residuals": True,
                    "third_order_residuals": True,
                    "symmetrize": feature_args["pharm_symmetrize"],
                }

                # Copy input arguments
                if feature_args.get("pharm_num_projections"):
                    kwargs["num_projections"] = feature_args["pharm_num_projections"]

                if feature_args.get("pharm_quantization_step"):
                    kwargs["quantization_step"] = feature_args["pharm_quantization_step"]

                if feature_args.get("pharm_T"):
                    kwargs["T"] = feature_args["pharm_T"]

                if PHARM_ORIGINAL == feature_type:
                    features_grouped = extract_pharm_original_features_from_file(**kwargs)

                elif PHARM_REVISITED == feature_type:
                    features_grouped = extract_pharm_revisited_features_from_file(**kwargs)

                features = flatten_single(features_grouped)

            elif SPAM == feature_type:
                features_grouped = extract_spam686_features_from_file(input_file)
                features = flatten_single(features_grouped)

            else:
                raise ValueError("Unknown feature type")

            # Write single-item batch
            writer.write({
                "filenames": [os.path.basename(input_file)],
                "features": np.expand_dims(features, axis=0),
            })

        except Exception as e:
            log.error(f"Error extracting features from image \"{input_file}\". Skipping this image.")
            log.error(e)

    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory containing the JPEG images")
    parser.add_argument("--output_dir", required=False, type=str, help="Where to store resulting features")
    parser.add_argument("--feature_type", required=True, type=str, choices=[DCTR, JRM, CC_JRM, GFR, PHARM_ORIGINAL, PHARM_REVISITED, SPAM], help="Type of features to extract")

    # Distribution of work
    parser.add_argument("--skip_num_images", type=int, help="Skip given number of images")
    parser.add_argument("--take_num_images", type=int, help="Take given number of images")

    # Optional args
    parser.add_argument("--output_suffix", type=str, help="Optional suffix to append to output file")

    # PHARM-specific args
    parser.add_argument("--pharm_num_projections", type=int, help="Number of random projection matrices")
    parser.add_argument("--pharm_quantization_step", type=float, help="Quantization step")
    parser.add_argument("--pharm_T", type=int, help="Truncation threshold")
    parser.add_argument("--pharm_disable_symmetrization", action="store_false", dest="pharm_symmetrize", help="Disable symmetrization")
    parser.set_defaults(pharm_symmetrize=True)

    # GFR-specific args
    parser.add_argument("--gfr_num_rotations", type=int, help="Number of rotations")
    parser.add_argument("--gfr_quantization_steps", type=float, nargs="*", help="4 quantization steps")
    parser.add_argument("--gfr_truncation_threshold", type=int, help="Truncation threshold")

    args = vars(parser.parse_args())

    # Log args
    log.info(args)

    # Validate input arguments
    assert os.path.exists(args["input_dir"]), "Given input directory does not exist"

    # Find input files
    input_files = glob(os.path.join(args["input_dir"], "*.jpeg")) + glob(os.path.join(args["input_dir"], "*.jpg"))
    input_files = sorted(input_files)

    assert len(input_files) > 0, "Found 0 JPEG images in the given input directory."

    # Concatenate output filename
    output_filename = f"{args['feature_type']}_features"

    # Output suffix
    if args["output_suffix"]:
        output_filename += "_" + args["output_suffix"]

    # Skip the given number of images
    if args["skip_num_images"] is not None:
        input_files = input_files[args["skip_num_images"]:]
        output_filename += f"_skip_num_images_{args['skip_num_images']}"

    # Take the given number of images
    if args["take_num_images"] is not None:
        input_files = input_files[:args["take_num_images"]]
        output_filename += f"_take_num_images_{args['take_num_images']}"

    # Concatenate output filepath
    output_filename += ".h5"
    output_dir = args["output_dir"]

    if output_dir is None:
        # If no output directory is given, use the input directory
        output_dir = args["input_dir"]

    output_filepath = os.path.join(output_dir, output_filename)

    extract_features(input_files=input_files, output_file=output_filepath, feature_type=args["feature_type"], feature_args=args)

    log.info(f"Stored features to \"{output_filepath}\"")
