Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

Installation and setup
----------------------

To use ``sealwatch``, first install it using pip:

.. code-block:: console

   $ pip3 install sealwatch

Import the package with

>>> import sealwatch as sw

For steganalysis on pixels, load the image using `pillow` and `numpy`.

>>> import numpy as np
>>> from PIL import Image
>>> spatial_cover = np.array(Image.open("cover.png"))

For JPEG steganalysis, load the DCT coefficients using our sister project `jpeglib`.

>>> import jpeglib
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> im_spatial = jpeglib.read_spatial("cover.jpeg")


.. note::

   ``sealwatch`` expects the DCT coefficients in 4D shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].
   If you use to 2D DCT representation (as used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   >>> y_4d = sw.tools.jpegio_to_jpeglib(y_2d)
   >>> y_2d = sw.tools.jpeglib_to_jpegio(y_4d)


Training a feature-based detector
---------------------------------

>>> # extract features
>>> f0, f1 = [], []
>>> for fname in glob(f'images/*.png'):
...      x0 = np.array(Image.open(fname).convert('L'))
...      f0.append(sw.spam.extract(x0))
...      x1 = cl.lsb.simulate(x0, .4, modify=cl.LSB_MATCHING, seed=12345)
...      f1.append(sw.spam.extract(x1))
>>>
>>> # train-test split
>>> f0, f1 = np.array(f0), np.array(f1)
>>> X = np.concatenate([f0, f1], axis=0)
>>> y = np.concatenate([
...      np.zeros(f0.shape[0]),
...      np.ones(f1.shape[0])], axis=0)
>>> from sklearn.model_selection import train_test_split
>>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.5, random_state=42)
>>> # train naive Bayes detector from scikit-learn
>>> from sklearn.naive_bayes import GaussianNB
>>> model = GaussianNB().fit(X_tr, y_tr)
>>>
>>> # save trained model
>>> import pickle
>>> pickle.dump(model, open('model.pickle', 'wb'))
>>>
>>> # evaluate model
>>> y_tr_pred, y_te_pred = model.predict(X_tr), model.predict(X_te)
>>> from sklearn.metrics import accuracy_score
>>> acc_tr, acc_te = accuracy_score(y_tr, y_tr_pred), accuracy_score(y_te, y_te_pred)
>>> print(f'accuracy: train {acc_tr} test {acc_te}')

The package also contains FLD detector, popular in steganalysis.
It is fully functional, tested against the Matlab.
However, its interface does not match sklearn at the moment.
This may change in one of the future releases.
For now, you can use it as follows.

>>> X_tr, X_te, y_tr, y_te = X[::2], X[1::2], y[::2], y[1::2]  # balanced training set
>>> trainer = sw.ensemble_classifier.FldEnsembleTrainer(
...      Xc=X_tr[y_tr == 0],
...      Xs=X_tr[y_tr == +1],  # same length as Xc
...      seed=42, verbose=0)
>>> model, _ = trainer.train()
>>>
>>> # save trained model ...
>>> # evaluate model
>>> y_tr_pred, y_te_pred = model.predict(X_tr), model.predict(X_te)
>>> acc_tr = accuracy_score(y_tr, (y_tr_pred + 1) / 2)  # {-1,-1} -> {0,1}
>>> acc_te = accuracy_score(y_te, (y_te_pred + 1) / 2)  # {-1,-1} -> {0,1}
>>> print(f'accuracy: train {acc_tr} test {acc_te}')


