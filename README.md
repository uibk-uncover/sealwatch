[![PyPI version](https://badge.fury.io/py/sealwatch.svg)](https://pypi.org/project/sealwatch/)
[![Commit CI/CD](https://github.com/uibk-uncover/sealwatch/actions/workflows/on_commit.yml/badge.svg)](https://github.com/uibk-uncover/sealwatch/actions/workflows/on_commit.yml)
[![Release CI/CD](https://github.com/uibk-uncover/sealwatch/actions/workflows/on_release.yml/badge.svg)](https://github.com/uibk-uncover/sealwatch/actions/workflows/on_release.yml)
[![Documentation Status](https://readthedocs.org/projects/sealwatch/badge/?version=latest)](https://sealwatch.readthedocs.io/)
[![PyPI downloads](https://img.shields.io/pypi/dm/sealwatch)](https://pypi.org/project/sealwatch/)
[![Stars](https://img.shields.io/github/stars/uibk-uncover/sealwatch.svg)](https://GitHub.com/uibk-uncover/sealwatch)
[![Contributors](https://img.shields.io/github/contributors/uibk-uncover/sealwatch)](https://GitHub.com/uibk-uncover/sealwatch)
[![Wheel](https://img.shields.io/pypi/wheel/sealwatch)](https://pypi.org/project/sealwatch/)
[![Status](https://img.shields.io/pypi/status/sealwatch)](https://pypi.com/project/sealwatch/)
<!-- [![PyPi license](https://badgen.net/pypi/license/sealwatch/)](https://pypi.com/project/sealwatch/) -->
[![Last commit](https://img.shields.io/github/last-commit/uibk-uncover/sealwatch)](https://GitHub.com/uibk-uncover/sealwatch)

<img src="https://raw.githubusercontent.com/uibk-uncover/sealwatch/main/docs/static/seal.png" width="300" />

# sealwatch

Python package, containing implementations of modern image steganalysis algorithms.

> :warning: This project is under intensive development as we speak.

## Installation

Simply install the package with pip3


```bash
pip3 install sealwatch
```

or using the cloned repository

```bash
git clone https://github.com/uibk-uncover/sealwatch/
cd sealwatch
pip3 install .
```

## Contents

| Abbreviation | Dimensionality | Domain | Reference | Output format |
|--------------|----------------|--------|----------:|---------------|
| SPAM: subtractive pixel adjacency matrix | 686 | spatial | [Reference](https://doi.org/10.1109/TIFS.2010.2045842) | ordered dict |
| JRM: JPEG rich model | 11255 | JPEG | [Reference](https://doi.org/10.1117/12.907495) | ordered dict |
| CC-JRM: cartesian-calibrated JRM | 22510 | JPEG | [Reference](https://doi.org/10.1117/12.907495) | ordered dict |
| DCTR: discrete cosine transform residual features | 8000 | spatial | [Reference](https://doi.org/10.1109/TIFS.2014.2364918) | 1D array |
| PHARM: phase-aware projection rich model | 12600 | JPEG | [Reference](https://doi.org/10.1117/12.2075239) | ordered dict |
| GFR: Gabor filter residual features | 17000 | JPEG | [Reference](https://dl.acm.org/doi/10.1145/2756601.2756608) | 5D array |
| SRM: spatial rich models | 34671 | spatial | [Reference](https://doi.org/10.1109/TIFS.2012.2190402) | ordered dict |
| SRMQ1: SRM with quantization 1 | 12753 | spatial | [Reference](https://doi.org/10.1109/TIFS.2012.2190402) | ordered dict |
| CRM: color rich models | 5404 | spatial | [Reference](https://doi.org/10.1109/WIFS.2014.7084325) | ordered dict |

These implementations are based on the [Matlab reference implementations](https://dde.binghamton.edu/download/feature_extractors/) provided by the DDE lab at Binghamton University.

## Usage

Extract features from a single JPEG image

```python
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file

features = extract_gfr_features_from_file("seal1.jpg")
```

Extract features for a directory of JPEG images and store them to a HDF5 file in the output directory.

```bash
python examples/batch_extraction/extract_features.py \
  --input_dir input_dir \
  --output_dir output_dir \
  --feature_type "gfr"
```

After having extracted features from cover and stego images, you can train an FLD ensemble as binary classifier.

```python
import numpy as np
from sealwatch.ensemble_classifier import FldEnsembleTrainer
from sealwatch.ensemble_classifier.helpers import load_and_split_features

cover_features = "cover_features.h5"
stego_features = "stego_features.h5"
train_csv = "train.csv"
test_csv = "test.csv"

cover_features_train, stego_features_train, cover_features_test, stego_features_test = load_and_split_features(
    cover_features_filename=cover_features,
    stego_features_filename=stego_features,
    train_csv=train_csv,
    test_csv=test_csv,
)

# Training is faster when arrays are C-contiguous
cover_features_train = np.ascontiguousarray(cover_features_train)
stego_features_train = np.ascontiguousarray(stego_features_train)

# The hyper-parameter search is wrapped inside a trainer class
trainer = FldEnsembleTrainer(
    Xc=cover_features_train,
    Xs=stego_features_train,
    seed=12345,
    verbose=1,
)

# Train with hyper-parameter search
trained_ensemble, training_records = trainer.train()

# Concatenate the test features and labels
X_test = np.concatenate((cover_features_test, stego_features_test), axis=0)
y_test = np.concatenate((
    -np.ones(len(cover_features_test)),
    +np.ones(len(stego_features_test))
), axis=0)

# Calculate test accuracy
test_accuracy = trained_ensemble.score(X_test, y_test)
```

### Feature formats

Note that the feature extractors return different formats: 1D arrays, multi-dimensional arrays, or ordered dicts.
The reason is that feature descriptors are composed of multiple submodels. Retaining the structure allows the user to select a specific submodel. The following snippets show how to flatten the features to a 1D array.


**Multi-dimensional array**
```python
from sealwatch.features.gfr import extract_gfr_features_from_file

# The GFR feature extraction returns a 5-dimensional array:
# - Dimension 0: Phase shifts
# - Dimension 1: Scales
# - Dimension 2: Rotations/Orientations
# - Dimension 3: Number of histograms
# - Dimension 4: Co-occurrences
features = extract_gfr_features_from_file("seal1.jpg")

# Simply flatten to a 1D array
features = features.flatten()
```

**Ordered dict**
```python
from sealwatch.features.pharm import extract_pharm_revisited_features_from_file
from sealwatch.utils.grouping import flatten_single

# The PHARM feature extraction returns an ordered dict
features_grouped = extract_pharm_revisited_features_from_file("seal1.jpg")

# Flatten dict to a 1D array
features = flatten_single(features_grouped)
```

After saving a batch of flattened features to an HDF5 file, you can also re-group them.
```python
from sealwatch.utils.grouping import group_batch
from sealwatch.utils.constants import PHARM_REVISITED
import h5py

# Load the flattened features
with h5py.File("pharm_features.h5", "r") as f:
  features_flat = f["features"][()]

# Re-group the flat features
features_grouped = group_batch(features_flat, feature_type=PHARM_REVISITED)

# features_grouped is an ordered dict. The keys are the submodel names. Each value is an array with the shape [num_samples, submodel_size].
```

```python
from sealwatch.utils.grouping import flatten_single

# PHARM feature extraction returns an ordered dict
features_grouped = extract_pharm_original_features_from_file(**kwargs)

# Flatten dict to a 1D ndarray
features = flatten_single(features_grouped)

# GFR feature extraction returns a 5D ndarray
features_5d = extract_gfr_features_from_file(**kwargs)

# Simply flatten the array
features = features.flatten()
```


## Acknowledgements and Disclaimer

Developed by [Martin Benes](https://github.com/martinbenes1996) and [Benedikt Lorch](https://github.com/btlorch/), University of Innsbruck, 2024.

The implementations of feature extractors and the detector in this package are based on the original Matlab code provided by the Digital Data Embedding Lab at Binghamton University.

We have made our best effort to ensure that our implementations produce identical results as the original Matlab implementations. However, it is the user's responsibility to verify this.
For notes on compatibility with previous implementation, see [compatibility.md](compatibility.md).
