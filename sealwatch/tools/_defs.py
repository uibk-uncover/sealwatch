"""

Author: Benedikt Lorch, Martin Benes
Affiliation University of Innsbruck
"""

import h5py
import logging
import numpy as np
import os
from pathlib import Path
import sys
from typing import Union


EPS = np.finfo(np.float64).eps
"""small numerical constant"""


def setup_custom_logger(
    name: Union[Path, str],
) -> logging.Logger:
    # Borrowed from https://stackoverflow.com/questions/7621897/python-logging-module-globally
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    #
    logger = logging.getLogger(str(name))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


class BufferedWriter(object):
    def __init__(self, output_filename, chunk_size=256, required_keys=()):
        """
        Initialize buffered writer
        :param output_filename: filepath where to store the results
        :type output_filename: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
        :param chunk_size: number of samples per chunk
        :type chunk_size: int
        :param required_keys: keys that must be given in each batch
        :type required_keys:

        :Example:

        >>> # TODO
        """

        if os.path.exists(output_filename):
            raise ValueError("Output file exists already")

        self._output_filename = output_filename
        self._chunk_size = chunk_size
        self._required_keys = required_keys

        # Buffering data structures for incremental writes
        self._cache_buffer = dict()

    @staticmethod
    def _count_samples(buffers_dict):
        """
        Counts the number of samples for each data buffer in the given dict
        :param buffers_dict: dict with different data sets
        :type buffers_dict:
        :return: dict containing the dataset names as keys and their sizes as values
        :rtype:
        """
        if 0 == len(buffers_dict):
            return {}

        return {key: len(buffers_dict[key]) for key in buffers_dict.keys()}

    def _add_to_cache(self, batch):
        """
        Add given batch to local cache buffer
        :param batch: dict
        :type batch:
        """
        for key, data in batch.items():

            # Does the key exist in our cache?
            if key not in self._cache_buffer.keys():
                # Create new item
                self._cache_buffer[key] = data

            else:
                # Concatenate with existing cache entries
                if isinstance(data, list) or isinstance(data, tuple):
                    self._cache_buffer[key] = self._cache_buffer[key] + data
                elif isinstance(data, np.ndarray):
                    self._cache_buffer[key] = np.concatenate([self._cache_buffer[key], data])
                else:
                    logging.error("Unexpected buffer type")
                    raise ValueError("Unexpected buffer type")

    def write(self, batch):
        """
        Writes the given batch to a cache buffer.
        Once the cache buffer reaches the predefined chunk size, multiples of the chunk size are written to the file.
        The remaining items are kept in the cache buffer.
        :param batch: dictionary with data sets as list or ndarray
        :type batch:
        """

        # Make sure that the batch contains at least all required keys
        for key in self._required_keys:
            assert key in batch, f"Given batch does not contain key {key}"

        # Copy data to cache
        self._add_to_cache(batch)

        # Count number of samples in the cache
        cached_dset_sizes = self._count_samples(self._cache_buffer)

        # Once the cache reaches the predefined chcunk sizes, multiples of the chunk size are written to file.
        while len(cached_dset_sizes) > 0 and min(cached_dset_sizes.values()) >= self._chunk_size:
            # Next chunk of data
            chunk = dict()

            # Copy dictionary because the original dictionary changes its size during the iteration
            for key, data in self._cache_buffer.copy().items():

                # Write full dataset, also if it exceeds the chunk size
                if len(data) >= self._chunk_size:

                    # Pop and clear cache entry
                    chunk[key] = self._cache_buffer.pop(key)

            # Write chunk to file
            self._write(chunk)

            # Repeat counting number of samples in dataset
            cached_dset_sizes = self._count_samples(self._cache_buffer)

    def _write(self, chunk):
        """
        Write chunk to the output file
        :param chunk: data already concatenated and ready to be written
        :type chunk:
        :return:
        :rtype:

        :Example:

        >>> # TODO
        """
        if 0 == len(chunk.keys()):
            logging.warning("Given chunk is empty")
            return

        # Append to the output file
        with h5py.File(self._output_filename, "a") as f:
            # Initialize a resizable data set to hold the output
            for key, data in chunk.items():
                chunk_size = len(data)

                # Append if dataset exists already
                if key in f.keys():
                    dataset = f[key]
                    current_dataset_size = len(dataset)

                    # Resize the data set to accommodate the next chunk of rows
                    dataset.resize(current_dataset_size + chunk_size, axis=0)

                    # Write the next chunk
                    dataset[current_dataset_size:] = data

                else:
                    # Create a new dataset
                    kwargs = {}
                    if isinstance(data, list) or isinstance(data, tuple):
                        shape = (chunk_size,)
                        maxshape = (None,)
                        # Select dtype based on data's type
                        if isinstance(data[0], (np.integer, int)):
                            dtype = int
                        elif isinstance(data[0], (np.floating, float)):
                            dtype = float
                        elif isinstance(data[0], str):
                            dtype = h5py.special_dtype(vlen=str)
                        else:
                            dtype = h5py.special_dtype(vlen=bytes)

                    elif isinstance(data, np.ndarray):
                        shape = data.shape
                        maxshape = (None,) + data.shape[1:]
                        dtype = data.dtype
                        kwargs["compression"] = "gzip"
                        kwargs["chunks"] = (self._chunk_size,) + data.shape[1:]
                    else:
                        logging.error("Unknown item type")

                    # Set up data set
                    dataset = f.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
                    dataset[:] = data

    def flush(self):
        if 0 == len(self._cache_buffer.keys()):
            # Nothing to write
            return

        self._write(self._cache_buffer)

        # After writing all the items, clear the buffer
        self._cache_buffer = dict()
