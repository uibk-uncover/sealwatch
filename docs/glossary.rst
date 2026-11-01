Glossary
========

Short definitions of the terms used in the documentation.
For the concepts behind them, see the :doc:`guide/index`,
and for JPEG specifically, the `glossary of jpeglib <https://jpeglib.readthedocs.io/en/latest/glossary.html>`__.

.. glossary::
   :sorted:

   analytical attack
      A detector derived from a model of a specific embedding,
      which estimates its :term:`payload` without training, e.g., WS.
      It is a :term:`targeted attack`.

   AUC
      The area under the :term:`ROC curve`, 1 for a perfect :term:`detector` and 0.5 for guessing.

   bpnzAC
      Bits per nonzero AC coefficient, the usual :term:`embedding rate` of JPEG steganography.

   bpp
      Bits per pixel, the usual :term:`embedding rate` of spatial steganography.

   change rate
      The share of elements changed by the embedding, denoted β.
      For :term:`LSB replacement`, half of the :term:`embedding rate`.

   cover
      The image before the embedding.

   DCT coefficient
      A coefficient of the discrete cosine transform of an 8x8 block of a JPEG image,
      quantized by the :term:`quantization table`.
      The first coefficient of each block is the DC, the other 63 are AC.

   detector
      A classifier deciding whether an image is :term:`cover` or :term:`stego`,
      e.g., the :term:`ensemble of FLD` on :term:`features <feature>`, or a neural network.

   EER
      The equal error rate, at the :term:`threshold` where the rates of :term:`false alarm`
      and :term:`missed detection` are equal.

   embedding rate
      The :term:`payload` relative to the size of the :term:`cover`, denoted α,
      e.g., in :term:`bpp` or :term:`bpnzAC`.

   ensemble of FLD
      A :term:`detector` of :term:`FLDs <FLD>`,
      each trained on a random subspace of the :term:`features <feature>`, voting by majority.

   false alarm
      A :term:`cover` detected as :term:`stego`.
      Its rate is denoted :math:`P_{FA}`, also the false positive rate.

   feature
      A vector describing an image for a :term:`detector`,
      e.g., SPAM or a :term:`rich model`, consisting of :term:`submodels <submodel>`.

   FLD
      Fisher linear discriminant, a linear classifier,
      which projects the :term:`features <feature>` onto the direction
      that best separates :term:`cover` and :term:`stego`,
      and thresholds the projection.

   hard prediction
      The class predicted by a :term:`detector`, 0 for :term:`cover` and 1 for :term:`stego`.

   LSB matching
      Embedding that changes a pixel by +1 or -1 at random,
      where its least significant bit does not match the message bit.

   LSB replacement
      Embedding that replaces the least significant bits of the pixels by the message bits.

   missed detection
      A :term:`stego` detected as :term:`cover`.
      Its rate is denoted :math:`P_{MD}`, also the false negative rate.

   P_E
      The minimal average error, the average of the rates of :term:`false alarm`
      and :term:`missed detection`, at the best :term:`threshold`.

   payload
      The length of the message, in bits.

   quantization table
      The 8x8 table of step sizes, by which a JPEG image divides its DCT coefficients.

   reference implementation
      The implementation of a method by its authors.
      sealwatch is tested to produce the same features or predictions.

   residual
      The difference between an image and its prediction from the neighboring pixels,
      in which the embedding changes are easier to spot.

   rich model
      A :term:`feature` set of many :term:`submodels <submodel>`,
      co-occurrences of quantized :term:`residuals <residual>`, e.g., SRM or JRM.

   ROC curve
      The rate of detected :term:`stego` against the rate of :term:`false alarm`,
      over all :term:`thresholds <threshold>`.

   soft prediction
      The score of a :term:`detector`, e.g., the stego probability.

   steganalysis
      The detection of steganography.

   stego
      The image after the embedding.

   submodel
      A part of a :term:`feature` set, e.g., from one :term:`residual`,
      returned as an entry of the dict of features.

   targeted attack
      :term:`Steganalysis <steganalysis>` designed for a specific embedding,
      which may not detect other embedding.

   threshold
      The score, at or above which a :term:`soft prediction` is predicted as :term:`stego`.

   wAUC
      The weighted :term:`AUC` of the ALASKA2 competition,
      which weights the part of the :term:`ROC curve` below the true positive rate of 0.4 twice.
