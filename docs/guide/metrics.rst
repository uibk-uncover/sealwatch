Evaluating a detector
=====================

The metrics are meters, which collect the labels and the predictions of a detector, e.g., batch by batch,
and compute the metric over all of them.
Get a meter by its name, and update it with the labels (0 for cover, 1 for stego) and the predictions,
e.g., of the detectors trained in :doc:`detector`.

.. tab-set::

   .. tab-item:: Ensemble of FLD

      .. code-block:: python

         y_pred = model.predict_confidence(X_te)  # in [-1, +1]

         meter = sw.metrics.get_meter("auc", backend="numpy")
         meter.update(y_te, y_pred)
         print(meter.avg)

   .. tab-item:: Neural network

      .. code-block:: python

         meter = sw.metrics.get_meter("auc", backend="torch")
         model.eval()
         with torch.no_grad():
             for x_b, y_b in loader:
                 y_pred = torch.softmax(model(x_b), dim=1)[:, 1]  # stego probability
                 meter.update(y_b, y_pred)
         print(meter.avg)

The backend ``"numpy"`` takes arrays, and ``"torch"`` takes tensors.
The torch meters detach the tensors, so that they don't keep the computational graph during training.
The detached tensors are kept on their device, and the average is a Python float.

.. warning::

   The detectors label cover and stego differently.
   The neural networks predict 0 for cover and 1 for stego,
   which arises naturally from the softmax activation.
   The ensembles predict -1 for cover and +1 for stego, which makes the voting of the FLD more natural.
   The ensemble of FLD predicts -1 for cover and +1 for stego, for backward compatibility,

   The meters are adaptive in the scale of soft predictions, but they expect labels and hard predictions to be 0 and 1.
   To convert predictions of the ensemble of FLD to 0 and 1, use ``(y_hat + 1) / 2``.


Soft predictions
----------------

Some metrics take soft predictions,
i.e., scores such as the stego probabilities,
and evaluate the detector over all thresholds.

- ``"auc"``, the area under the ROC curve,
- ``"wauc"``, the weighted area under the ROC curve, introduced in the ALASKA2 competition,
  exactly as there with ``implementation=sw.WAUC_ALASKA``,
- ``"fnr_at_fpr"``, the missed detection rate at 5% false alarms,
- ``"eer"``, the equal error rate, where the false alarm and the missed detection rate are equal.

For a neural network, the soft prediction is the stego probability.

.. code-block:: python

   logits = model(x)  # [B, 2]
   y_pred = torch.softmax(logits, dim=1)[:, 1]  # stego probability

   meter = sw.metrics.get_meter("auc", backend="torch")
   meter.update(y, y_pred)


Hard predictions
----------------

Other metrics take hard predictions, i.e., 0 for cover and 1 for stego,
and compare them with the labels.

- ``"accuracy"``, the rate of correct predictions,
- ``"misclassification"``, the rate of wrong predictions,
- ``"precision"``, the rate of stego among the images predicted as stego,
- ``"recall"``, the rate of stego images predicted as stego.

For a neural network, the hard prediction is the class with the larger logit.

.. code-block:: python

   logits = model(x)  # [B, 2]
   y_hat = logits.argmax(dim=1)  # 0 for cover, 1 for stego

   meter = sw.metrics.get_meter("accuracy", backend="torch")
   meter.update(y, y_hat)

Alternatively, give the soft predictions with a threshold,
at or above which an image is predicted as stego.

.. code-block:: python

   meter = sw.metrics.get_meter("accuracy", backend="torch", threshold=.5)
   meter.update(y, y_pred)  # stego probability


A note on :math:`P_E`
---------------------

The minimal average error :math:`P_E`, ``"pe"``, is the most common metric in steganalysis.
It takes soft predictions,
but it is the error rate of hard predictions:
the average of the false alarm and the missed detection rate, at the best threshold,

.. math::

   P_E = \min_\tau \frac{P_{FA}(\tau) + P_{MD}(\tau)}{2}.

:math:`P_E` always calibrates the threshold to be the best possible, on the data given to the meter.
This is a convention in research, which makes the detectors comparable,
but it is not realistic in practice,
where the threshold must be set before the labels of the test set are known.
For a realistic error, calibrate the threshold on the training set,
and carry it over to the test set, without calibrating it there.

.. code-block:: python

   # calibrate the threshold on the training set
   meter = sw.metrics.get_meter("pe", backend="torch")
   meter.update(y_tr, y_pred_tr)  # calibrates the threshold
   tau = meter.tau

   # test set, at the threshold of the training set
   meter = sw.metrics.get_meter("pe", backend="torch", threshold=tau)
   meter.update(y_te, y_pred_te)  # uses a fixed threshold
   print(meter.avg)
