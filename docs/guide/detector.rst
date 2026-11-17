Training a detector
===================

A feature-based detector extracts features from cover and stego images,
and trains a classifier on them.
A neural network learns from the images directly.


Extracting features
-------------------

Extract features from covers, and from stegos simulated with `conseal <https://conseal.readthedocs.io/>`__.

.. code-block:: python

   from glob import glob
   import conseal as cl
   import numpy as np
   from PIL import Image
   import sealwatch as sw

   # feature extraction
   f0, f1 = [], []
   for fname in glob("images/*.png"):
       x0 = np.array(Image.open(fname).convert("L"))
       f0.append(sw.tools.features.flatten(sw.spam.extract(x0)))
       x1 = cl.lsb.simulate(x0, .4, modify=cl.LSB_MATCHING, seed=12345)
       f1.append(sw.tools.features.flatten(sw.spam.extract(x1)))

   # prepare dataset
   f0, f1 = np.array(f0), np.array(f1)
   X = np.concatenate([f0, f1], axis=0)
   y = np.concatenate([
       np.zeros(f0.shape[0]),
       np.ones(f1.shape[0]),
   ], axis=0)


Ensemble of FLD
---------------

The ensemble of FLD [Kodovsky2012]_ is the classifier popular in steganalysis.
It follows the scikit-learn interface, and matches the Matlab reference implementation.

The ensemble assumes pairs: the training set holds each cover together with its stego,
covers first and stegos in the same order.
Therefore, split the images, not the samples.

.. code-block:: python

   # random split of the images, cover and stego of an image together
   rng = np.random.default_rng(12345)
   idx = rng.permutation(len(f0))
   idx_tr, idx_te = idx[:len(idx) // 2], idx[len(idx) // 2:]
   X_tr = np.concatenate([f0[idx_tr], f1[idx_tr]])
   X_te = np.concatenate([f0[idx_te], f1[idx_te]])
   y_tr = np.concatenate([np.zeros(len(idx_tr)), np.ones(len(idx_tr))])
   y_te = np.concatenate([np.zeros(len(idx_te)), np.ones(len(idx_te))])

   # train
   model = sw.ensemble_classifier.FldEnsembleClassifier(random_state=42)
   model.fit(X_tr, y_tr)

   # evaluate, predictions are in {0, 1}
   acc_tr = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_tr.update(y_tr, model.predict(X_tr))
   acc_te = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_te.update(y_te, model.predict(X_te))
   print(f"accuracy: train {acc_tr.avg} test {acc_te.avg}")

The subspace dimensionality ``d_sub`` and the number of base learners ``L`` are
found automatically, from the out-of-bag error estimate on the training set.
The search over ``d_sub`` trains an ensemble for each candidate, which is costly on large features such as SRM.
It can be sped up, or skipped.

.. code-block:: python

   # cheaper search: at most 50 base learners per d_sub candidate
   model = sw.ensemble_classifier.FldEnsembleClassifier(search_L=50, random_state=42)

   # no search
   model = sw.ensemble_classifier.FldEnsembleClassifier(d_sub=1000, random_state=42)

``model.decision_function(X)`` gives the confidence in the range from -1 (cover) to +1 (stego),
the vote of the base learners.
``model.training_records_`` lists the ``d_sub`` candidates, with their out-of-bag errors.

.. note::

   Until release 2026.11, the ensemble was trained with ``FldEnsembleTrainer``,
   on separate covers and stegos, with labels -1 and +1.
   This interface still works, with a deprecation warning, and gives the same results.
   To migrate, concatenate the covers and the stegos and pass them to ``fit``.

   .. code-block:: python

      # before
      model, _ = sw.ensemble_classifier.FldEnsembleTrainer(Xc=Xc, Xs=Xs, seed=42).train()

      # now
      X = np.concatenate([Xc, Xs])
      y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xs))])
      model = sw.ensemble_classifier.FldEnsembleClassifier(random_state=42).fit(X, y)

.. [Kodovsky2012] J. Kodovský, J. Fridrich, V. Holub.
   Ensemble classifiers for steganalysis of digital media.
   IEEE TIFS, 7(2):432–444, 2012.


scikit-learn classifier
-----------------------

Alternatively, any classifier from scikit-learn can be trained on the features,
e.g., naive Bayes.

.. code-block:: python

   import pickle
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import GaussianNB

   # train-test split
   X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.5, random_state=42)

   # train
   model = GaussianNB().fit(X_tr, y_tr)

   # save trained model
   pickle.dump(model, open("model.pickle", "wb"))

   # evaluate, predictions are in {0, 1}
   acc_tr = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_tr.update(y_tr, model.predict(X_tr))
   acc_te = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_te.update(y_te, model.predict(X_te))
   print(f"accuracy: train {acc_tr.avg} test {acc_te.avg}")


Neural network
--------------

A neural network, such as XuNet, is trained with `PyTorch <https://pytorch.org/>`__,
on images of the same size.
The images are loaded per batch, and the stego images are simulated on the fly.

.. code-block:: python

   import torch

   # cover and stego of each image, with labels 0 and 1
   fnames = glob("images/*.png")
   items = [(i, fname, label) for i, fname in enumerate(fnames) for label in (0, 1)]

   def prepare(item):
       i, fname, label = item
       x = np.array(Image.open(fname).convert("L"))
       if label == 1:
           x = cl.lsb.simulate(x, .4, modify=cl.LSB_MATCHING, seed=12345 + i)  # same stego in each epoch
       return torch.from_numpy(x.astype("float32") / 255.)[None], label  # [C, H, W] in [0, 1]

   def collate(batch):
       return torch.utils.data.default_collate([prepare(item) for item in batch])

   loader = torch.utils.data.DataLoader(items, batch_size=16, shuffle=True, collate_fn=collate)

   # train
   model = sw.nn.XuNet(in_channels=1, num_classes=2, weight_seed=12345)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   model.train()
   for epoch in range(10):
       for x_b, y_b in loader:
           optimizer.zero_grad()
           loss = torch.nn.functional.cross_entropy(model(x_b), y_b)
           loss.backward()
           optimizer.step()

   # save trained model
   torch.save(model.state_dict(), "xunet.pt")

The trained model is loaded as pretrained, and predicts a single image.

.. code-block:: python

   model = sw.nn.XuNet.pretrained("xunet.pt")
   x1 = np.array(Image.open("suspicious.png").convert("L"))
   y_pred = sw.nn.infere_single(x1, model)  # stego probability
