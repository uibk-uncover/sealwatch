import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import h5py
import os
import re
from sealwatch.utils.logger import setup_custom_logger
from sealwatch.utils.writer import BufferedWriter


log = setup_custom_logger(os.path.basename(__file__))


def merge_h5_files(batch_files, output_file, chunk_size=256):
    """
    Concatenate features from individual HDF5 files to one HDF5 file
    :param batch_files: list of HDF5 files
    :param output_file: path to output HDF5 file
    :param chunk_size: chunk size
    """
    # Set up buffered writer
    writer = BufferedWriter(output_file, chunk_size=chunk_size)

    # Iterate over individual batch files
    for batch_file in tqdm(batch_files):
        # Read batch file
        with h5py.File(batch_file, "r") as f:
            # Determine number of samples
            num_samples = None
            for key in f.keys():
                if num_samples is None:
                    num_samples = len(f[key])
                else:
                    assert num_samples == len(f[key]), "Expected same number of samples in the dataset"

            # Load and write batches
            num_batches = int(np.ceil(num_samples / chunk_size))
            for batch_idx in range(num_batches):
                batch_from = batch_idx * chunk_size
                batch_to = min(num_samples, (batch_idx + 1) * chunk_size)

                # Load batch
                batch = {}
                for key in f.keys():
                    if key == "filenames":
                        # Special treatment for string filenames
                        filenames = f[key][batch_from:batch_to]
                        filenames = [filename.decode("utf-8") for filename in filenames]
                        batch[key] = filenames
                    else:
                        batch[key] = f[key][batch_from:batch_to]

                # Write batch
                writer.write(batch)

    # Flush and close the buffered writer
    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Input directory where to look for HDF5 input files")
    parser.add_argument("--feature_type", required=True, type=str, help="Locate filenames that start with this feature type")

    # Optional
    parser.add_argument("--output_suffix", type=str, help="Append suffix to the output filename")

    args = vars(parser.parse_args())

    # Validate input arguments
    assert os.path.exists(args["input_dir"]), "Given input directory does not exist"

    # Locate batch files
    input_glob = os.path.join(args["input_dir"], args["feature_type"] + "_*.h5")
    batch_files = glob(input_glob)

    assert len(batch_files) > 0, f"No files found with the given pattern \"{input_glob}\"."

    # Log which files were found
    log.info(f"Found {len(batch_files)} input files:")
    for batch_file in batch_files:
        log.info(os.path.relpath(batch_file, args["input_dir"]))

    # Sort batch files by the number of images skipped
    num_images_skipped = []
    for batch_file in batch_files:
        match = re.search("skip_num_images_([0-9]+)", batch_file)
        if match is None:
            num_images_skipped.append(0)
        else:
            num_images_skipped.append(int(match.group(1)))

    permutation = np.argsort(num_images_skipped)
    batch_files = [batch_files[i] for i in permutation]

    # Try to derive feature name from input files
    match = re.search("([A-Za-z0-9_.]+)_skip_num_images_[0-9]+_take_num_images_[0-9]+.h5", os.path.basename(batch_files[-1]))
    assert match is not None, "Input file does not match expected pattern"

    # Construct output filepath
    output_filename = match.group(1)

    # Output suffix
    if args["output_suffix"]:
        output_filename += "_" + args["output_suffix"]

    output_filename += ".h5"
    output_filepath = os.path.join(args["input_dir"], output_filename)

    # Merge HDF5 files
    merge_h5_files(batch_files, output_filepath)

    # Log output path
    log.info(f"Stored results to \"{output_filepath}\"")
