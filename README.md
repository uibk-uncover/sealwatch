TODOs:
- Consistent flattening/grouping of features
- Add license headers

# Usage

Extract features from a single JPEG image

```python
from sealwatch.features.gfr.gfr import extract_gfr_features_from_file

features = extract_gfr_features_from_file("seal1.jpg")
```

Extract features for a directory of JPEG images

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
test_accuracy = trained_ensemble.score(cover_features_test, y_test)
```

# Unit tests

To ensure that our features are equivalent to the features produced by the Matlab, we slightly altered the original Matlab implementations.
In particular, we replaced the rounding operator and methods that draw random numbers with a custom implementation that can be reproduced in Python.
For more details, see [this blog post](https://www.benediktlorch.com/blog/2023/matlab-to-python/).

## Other differences

- The Matlab implementation of the PHARM features contains a bug in the symmetrization. We provide both the original implementation (*pharm_original*) and a variant with correct symmetrization (*pharm_revisited*). The revisited implementation also does not crop the image borders to simplify the indexing. Therefore, it gives slightly different results than the original implementation when symmetrization is disabled.
- The Matlab implementation of the FLD ensemble contains two peculiarities:
  * The function `add_gridpoints` in lines 588-605 looks like it implements insertion into a sorted list, but it inserts the new element one position too early.
  * While searching for the optimal `d_sub`, the condition in line 355 uses `settings.d_sub`. However, `settings.d_sub` is the `d_sub` that was previously evaluated. We believe that `settings.d_sub` should be replaced by `SEARCH.x(minE_id)` in line 355 and in line 358.