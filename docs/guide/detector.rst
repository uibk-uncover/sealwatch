Training a detector
===================

A feature-based detector extracts features from cover and stego images,
and trains a classifier on them.


Extracting features
-------------------

Extract features from covers, and from stegos simulated with `conseal <https://conseal.readthedocs.io/>`__.

.. code-block:: python

   from glob import glob
   import conseal as cl
   import numpy as np
   from PIL import Image
   import sealwatch as sw

   f0, f1 = [], []
   for fname in glob("images/*.png"):
       x0 = np.array(Image.open(fname).convert("L"))
       f0.append(sw.tools.features.flatten(sw.spam.extract(x0)))
       x1 = cl.lsb.simulate(x0, .4, modify=cl.LSB_MATCHING, seed=12345)
       f1.append(sw.tools.features.flatten(sw.spam.extract(x1)))

   f0, f1 = np.array(f0), np.array(f1)
   X = np.concatenate([f0, f1], axis=0)
   y = np.concatenate([
       np.zeros(f0.shape[0]),
       np.ones(f1.shape[0]),
   ], axis=0)


scikit-learn classifier
-----------------------

Any classifier from scikit-learn can be trained on the features,
e.g., naive Bayes.

.. code-block:: python

   import pickle
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import GaussianNB

   # train-test split
   X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.5, random_state=42)

   # train
   model = GaussianNB().fit(X_tr, y_tr)

   # save trained model
   pickle.dump(model, open("model.pickle", "wb"))

   # evaluate
   y_tr_pred, y_te_pred = model.predict(X_tr), model.predict(X_te)
   acc_tr, acc_te = accuracy_score(y_tr, y_tr_pred), accuracy_score(y_te, y_te_pred)
   print(f"accuracy: train {acc_tr} test {acc_te}")


Ensemble of FLD
---------------

The package also contains the ensemble of FLD, popular in steganalysis.
It is fully functional, tested against the Matlab.
However, its interface does not match sklearn at the moment.
This may change in one of the future releases.
For now, you can use it as follows.

.. code-block:: python

   # balanced training set
   X_tr, X_te, y_tr, y_te = X[::2], X[1::2], y[::2], y[1::2]

   # train
   trainer = sw.ensemble_classifier.FldEnsembleTrainer(
       Xc=X_tr[y_tr == 0],
       Xs=X_tr[y_tr == +1],  # same length as Xc
       seed=42,
       verbose=0,
   )
   model, _ = trainer.train()

   # evaluate, predictions are in {-1, +1}
   y_tr_pred, y_te_pred = model.predict(X_tr), model.predict(X_te)
   acc_tr = accuracy_score(y_tr, (y_tr_pred + 1) / 2)
   acc_te = accuracy_score(y_te, (y_te_pred + 1) / 2)
   print(f"accuracy: train {acc_tr} test {acc_te}")
