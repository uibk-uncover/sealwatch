"""Feature store in zarr, written in parallel by the tasks of a job array.

Layout: one array of shape ``[num_samples, num_features]`` with the filenames of the samples,
as in the HDF5 feature files. One shard (= one file) per task, so that tasks running at the
same time never write into the same file. A sample that was not written yet holds NaN.
The store is a folder with one file per task and ``zarr.json``.

Typical use with SLURM::

    # once, before submitting
    sw.tools.create_feature_store('srm.zarr', filenames, num_features=34671, samples_per_task=100)
    # in each array task (sbatch --array=0-99)
    rows = sw.tools.feature_task_rows('srm.zarr', task)
    sw.tools.write_feature_task('srm.zarr', task, extract_all(filenames[rows.start:rows.stop]))
    # once, after the array job
    sw.tools.missing_feature_tasks('srm.zarr')  # tasks to rerun, if any
    # anywhere, like load_hdf5
    features, filenames = sw.tools.load_zarr('srm.zarr')

The loaders of paired features, such as :func:`~sealwatch.tools.load_features`,
take zarr stores as well as HDF5 files.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from __future__ import annotations

import os
import pathlib
import typing

import numpy as np

PathLike = typing.Union[str, os.PathLike]


def _zarr():
    """Import zarr on first use; the import takes about 0.3 s."""
    import zarr
    return zarr


def _open(store: PathLike, mode: str = 'r'):
    """Open the array of a store."""
    return _zarr().open_array(pathlib.Path(store), mode=mode)


def create_feature_store(
    store: PathLike,
    filenames: typing.Sequence[str],
    num_features: int,
    samples_per_task: int,
    *,
    samples_per_chunk: int = 10,
    dtype: str = 'float32',
    compression_level: int = 3,
):
    """Create an empty feature store. Run once, before the tasks.

    :param store: path of the store folder; must not exist yet.
    :param filenames: filenames of the samples (images), one per row, in the order of the rows.
    :param num_features: feature dimensionality, the columns.
    :param samples_per_task: rows written by one task, one shard file each.
    :param samples_per_chunk: rows compressed together; the unit of reading.
        Must divide ``samples_per_task``.
    :param dtype: floating-point type of the features.
    :param compression_level: zstd level.
    :return: the created zarr array.
    :raises ValueError: if ``samples_per_chunk`` does not divide ``samples_per_task``.
    :raises zarr.errors.ContainsArrayError: if the store exists already.
    """
    zarr = _zarr()
    if samples_per_task % samples_per_chunk:
        raise ValueError(f'samples_per_chunk ({samples_per_chunk}) must divide samples_per_task ({samples_per_task})')
    z = zarr.create_array(
        pathlib.Path(store),
        shape=(len(filenames), num_features),
        shards=(samples_per_task, num_features),  # one file per task
        chunks=(samples_per_chunk, num_features),
        dtype=dtype,
        fill_value=np.nan,  # marks rows not written yet
        compressors=zarr.codecs.ZstdCodec(level=compression_level),
        chunk_key_encoding={'name': 'default', 'separator': '.'},  # flat files, no subfolder per task
        overwrite=False,  # never wipe finished results
    )
    z.attrs['samples_per_task'] = samples_per_task
    z.attrs['filenames'] = [f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in filenames]
    return z


def num_feature_tasks(store: PathLike) -> int:
    """Number of tasks of a store, i.e., the size of the job array."""
    z = _open(store)
    return -(-z.shape[0] // z.attrs['samples_per_task'])


def feature_task_rows(store: PathLike, task: int) -> range:
    """Rows written by a task.

    :param store: path of the store.
    :param task: task index, e.g., ``$SLURM_ARRAY_TASK_ID``.
    :return: range of the rows (sample indices) of the task.
    :raises IndexError: if the task does not exist.
    """
    z = _open(store)
    per_task = z.attrs['samples_per_task']
    if not 0 <= task < -(-z.shape[0] // per_task):
        raise IndexError(f'Task {task} out of range')
    return range(task * per_task, min((task + 1) * per_task, z.shape[0]))


def write_feature_task(store: PathLike, task: int, features: np.ndarray) -> None:
    """Write the features of a task, its whole shard at once.

    :param store: path of the store folder.
    :param task: task index.
    :param features: features of the task's rows, of shape ``[len(feature_task_rows(store, task)), num_features]``.
    :raises ValueError: if the shape does not match, or a feature is not finite
        (NaN marks missing rows).
    """
    z = _open(store, mode='r+')
    rows = feature_task_rows(store, task)
    features = np.asarray(features)
    if features.shape != (len(rows), z.shape[1]):
        raise ValueError(f'Expected features of shape {(len(rows), z.shape[1])}, got {features.shape}')
    if not np.all(np.isfinite(features)):
        raise ValueError('Features must be finite')
    z[rows.start:rows.stop] = features


def missing_feature_tasks(store: PathLike) -> np.ndarray:
    """Tasks that did not write their rows, e.g., failed ones, to run again.

    :param store: path of the store.
    :return: indices of the missing tasks, empty when all are done.
    """
    z = _open(store)
    per_task = z.attrs['samples_per_task']
    missing = []
    for task, start in enumerate(range(0, z.shape[0], per_task)):
        try:
            if np.isnan(z[start, 0]):
                missing.append(task)
        except Exception:  # shard cut short, e.g., the task was killed while writing
            missing.append(task)
    return np.array(missing, dtype=int)


def load_zarr(
    features_filename: PathLike,
    max_num_samples: typing.Optional[int] = None,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Retrieve features and filenames from a feature store, as :func:`~sealwatch.tools.load_hdf5`.

    Rows of missing tasks are NaN.

    :param features_filename: path to the store
    :param max_num_samples: Only load the first n features and filenames
    :return: (features, filenames) as 2-tuple
    :raises ValueError: if a shard is cut short; see :func:`missing_feature_tasks`.
    """
    z = _open(features_filename)
    rows = slice(None, max_num_samples)
    try:
        features = z[rows]
    except Exception as e:
        raise ValueError(f'{features_filename} has damaged tasks, see missing_feature_tasks') from e
    filenames = np.array(z.attrs['filenames'])[rows]
    return features, filenames
