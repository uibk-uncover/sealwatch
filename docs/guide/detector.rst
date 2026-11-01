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

The ensemble of FLD is the classifier popular in steganalysis.
It is fully functional, tested against the Matlab.
However, its interface does not match sklearn at the moment.
This may change in one of the future releases.
For now, you can use it as follows.

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
   trainer = sw.ensemble_classifier.FldEnsembleTrainer(
       Xc=f0[idx_tr],
       Xs=f1[idx_tr],  # stego of the covers, in the same order
       seed=42,
       verbose=0,
   )
   model, _ = trainer.train()

   # evaluate, predictions are in {-1, +1}, the meters take {0, 1}
   acc_tr = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_tr.update(y_tr, (model.predict(X_tr) + 1) / 2)
   acc_te = sw.metrics.get_meter("accuracy", backend="numpy")
   acc_te.update(y_te, (model.predict(X_te) + 1) / 2)
   print(f"accuracy: train {acc_tr.avg} test {acc_te.avg}")


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
