# Contents

The following steganalysis feature implementations are provided:

| Abbreviation | Full name                                   | Dimensionality | Reference                                                   | Output format |
|--------------|---------------------------------------------|---------------:|-------------------------------------------------------------|-------------|
| SPAM         | Subtractive pixel adjacency matrix          |            686 | [Reference](https://doi.org/10.1109/TIFS.2010.2045842)      | ordered dict |
| JRM          | JPEG rich model                             |          11255 | [Reference](https://doi.org/10.1117/12.907495)              | ordered dict |
| DCTR         | Discrete cosine transform residual features |           8000 | [Reference](https://doi.org/10.1109/TIFS.2014.2364918)      | 1D array    |
| PHARM        | Phase-aware projection rich model           |          12600 | [Reference](https://doi.org/10.1117/12.2075239)             | ordered dict |
| GFR          | Gabor filter residual features              |          17000 | [Reference](https://dl.acm.org/doi/10.1145/2756601.2756608) | 5D array  |

These implementations are based on the [Matlab reference implementations](https://dde.binghamton.edu/download/feature_extractors/) provided by the DDE lab at Binghamton University.

# Usage

Extract features from a single JPEG image

```python
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file

features = extract_gfr_features_from_file("seal1.jpg")
```

Extract features for a directory of JPEG images and store them to a HDF5 file in the output directory.

```bash
python sealwatch/batch_extraction/extract_features.py \
  --input_dir input_dir \
  --output_dir output_dir \
  --feature_type "gfr"  
```

After having extracted features from cover and stego images, you can train an FLD ensemble as binary classifier.

```python
import numpy as np
from sealwatch.ensemble_classifier.fld_ensemble_trainer import FldEnsembleTrainer
from sealwatch.ensemble_classifier.load_features import load_and_split_features

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

## Feature formats

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

# Unit tests and compatibility

For notes on compatibility with previous implementation, see [compatibility.md](compatibility.md).
