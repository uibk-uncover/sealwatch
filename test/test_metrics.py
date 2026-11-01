"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import functools
import importlib.util
import logging
import numpy as np
import os
from parameterized import parameterized
import sealwatch as sw
import sys
import tempfile
import torch
from typing import List
import unittest
import urllib.request
import warnings

try:
    from sklearn import metrics as skm
except ImportError:
    skm = None


BACKENDS = {
    'numpy': np.asarray,
    'torch': lambda a: torch.as_tensor(np.asarray(a)),
}


def scores_and_labels(n: int = 1000, loc: float = .75, ties: bool = False, seed: int = 12345):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(size=n), rng.normal(loc=loc, size=n)])
    if ties:
        scores = np.round(scores, 1)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return labels, scores


def compute(name: str, backend: str, *ys, **kw):
    meter = sw.metrics.get_meter(name, backend, **kw)
    meter.update(*[BACKENDS[backend](y) for y in ys])
    return meter.avg


class TestMetrics(unittest.TestCase):
    """Test suite for metrics."""
    _logger = logging.getLogger(__name__)

    def test_register(self):
        self._logger.info('TestMetrics.test_register()')
        self.assertIs(sw.metrics.METER_REGISTRY[('acc', 'numpy')], sw.metrics.METER_REGISTRY[('accuracy', 'numpy')])
        with self.assertRaises(KeyError):
            sw.metrics.register_meter(name='other', backend='numpy', name_abbr='acc')(type('X', (), {}))
        with self.assertRaises(ValueError):
            sw.metrics.get_meter('unknown')

    @parameterized.expand([[b] for b in BACKENDS])
    def test_accuracy(self, backend: str):
        self._logger.info(f'TestMetrics.test_accuracy({backend=})')
        y_true = [1, 0, 1, 0, 0, 1]
        y_pred = [1, 0, 1, 0, 1, 0]
        self.assertAlmostEqual(compute('acc', backend, y_true, y_pred), 4/6)
        self.assertAlmostEqual(compute('misc', backend, y_true, y_pred), 2/6)
        self.assertAlmostEqual(compute('precision', backend, y_true, y_pred), 2/3)
        self.assertAlmostEqual(compute('recall', backend, y_true, y_pred), 2/3)
        # scores with threshold
        scores = [.9, .1, .8, .2, .7, .3]
        self.assertAlmostEqual(compute('acc', backend, y_true, scores, threshold=.5), 4/6)
        # undefined
        self.assertTrue(np.isnan(compute('precision', backend, [1., 0.], [0., 0.])))

    @parameterized.expand([
        (backend, ties)
        for backend in BACKENDS
        for ties in [False, True]
    ])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_roc_metrics(self, backend: str, ties: bool):
        """Exact against scikit-learn."""
        self._logger.info(f'TestMetrics.test_roc_metrics({backend=}, {ties=})')
        labels, scores = scores_and_labels(ties=ties)
        fpr, tpr, _ = skm.roc_curve(labels, scores)
        # AUC
        self.assertAlmostEqual(compute('auc', backend, labels, scores), skm.roc_auc_score(labels, scores))
        # P_E
        self.assertAlmostEqual(compute('pe', backend, labels, scores), np.min(.5 * (fpr + 1 - tpr)))
        # FNR at FPR
        level = .05
        fnr_below = 1 - np.max(tpr[fpr <= level])
        self.assertAlmostEqual(compute('fnr_fpr', backend, labels, scores, strategy='below'), fnr_below)
        fnr_interp = compute('fnr_fpr', backend, labels, scores)
        self.assertLessEqual(fnr_interp, fnr_below + 1e-12)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_given_pe(self, backend: str):
        self._logger.info(f'TestMetrics.test_given_pe({backend=})')
        labels, scores = scores_and_labels()
        meter = sw.metrics.get_meter('pe', backend)
        tau = .4
        expected = .5 * (np.mean(scores[labels == 0] >= tau) + 1 - np.mean(scores[labels == 1] >= tau))
        p_e, _ = meter.given_pe(BACKENDS[backend](labels), BACKENDS[backend](scores), tau)
        self.assertAlmostEqual(p_e, expected)

    @parameterized.expand([
        (backend, name)
        for backend in BACKENDS
        for name in ['acc', 'precision', 'pe', 'auc', 'fnr_fpr']
    ])
    def test_undefined(self, backend: str, name: str):
        """Empty or single-class meters return NaN."""
        self._logger.info(f'TestMetrics.test_undefined({backend=}, {name=})')
        meter = sw.metrics.get_meter(name, backend)
        self.assertTrue(np.isnan(meter.avg))
        if name in ['pe', 'auc', 'fnr_fpr']:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.assertTrue(np.isnan(compute(name, backend, [1., 1.], [.2, .8])))

    @parameterized.expand([[b] for b in BACKENDS])
    def test_batches(self, backend: str):
        """Accumulating batches equals computing at once."""
        self._logger.info(f'TestMetrics.test_batches({backend=})')
        labels, scores = scores_and_labels()
        for name in ['auc', 'pe', 'acc']:
            kw = {'threshold': 0.} if name == 'acc' else {}
            meter = sw.metrics.get_meter(name, backend, **kw)
            for i in range(0, len(labels), 300):
                meter.update(BACKENDS[backend](labels[i:i+300]), BACKENDS[backend](scores[i:i+300]))
            self.assertAlmostEqual(meter.avg, compute(name, backend, labels, scores, **kw), places=6)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_mean(self, backend: str):
        """Loss and MSE are means, weighted by the batch size."""
        self._logger.info(f'TestMetrics.test_mean({backend=})')
        meter = sw.metrics.get_meter('loss', backend)
        meter.update(BACKENDS[backend](1.), n=3)
        meter.update(BACKENDS[backend](5.), n=1)
        self.assertAlmostEqual(meter.avg, 2.)
        # MSE over unequal batches
        y_true = np.arange(10, dtype='float32')
        y_pred = y_true + np.arange(10, dtype='float32') % 3
        meter = sw.metrics.get_meter('mse', backend)
        meter.update(BACKENDS[backend](y_true[:8]), BACKENDS[backend](y_pred[:8]))
        meter.update(BACKENDS[backend](y_true[8:]), BACKENDS[backend](y_pred[8:]))
        self.assertAlmostEqual(meter.avg, np.mean((y_true - y_pred)**2), places=5)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_psnr(self, backend: str):
        self._logger.info(f'TestMetrics.test_psnr({backend=})')
        meter = sw.metrics.get_meter('psnr', backend, max_val=1.)
        self.assertEqual(meter.objective, 'maximize')
        meter.update(BACKENDS[backend](np.zeros(4)), BACKENDS[backend](np.full(4, .1)))
        self.assertAlmostEqual(meter.avg, 20., places=4)

    def test_scalar(self):
        self._logger.info('TestMetrics.test_scalar()')
        meter = sw.metrics.get_meter('scalar', 'numpy', key='l')
        meter.update(l=np.array([1., 1., 1.]))
        meter.update(l=np.array([5.]))
        self.assertAlmostEqual(meter.avg, 2.)
        meter = sw.metrics.get_meter('scalar', 'torch', key='l')
        meter.update(l=torch.tensor([1., 1., 1.]))
        meter.update(l=torch.tensor([5.]))
        self.assertAlmostEqual(meter.avg, 2.)

    def test_torch_to(self):
        """Meters move with .to(), including their state."""
        self._logger.info('TestMetrics.test_torch_to()')
        meter = sw.metrics.get_meter('auc', 'torch')
        meter.update(torch.tensor([0., 1.]), torch.tensor([.2, .8]))
        self.assertIs(meter.to('cpu'), meter)
        # scores are kept as given, and computed in float64
        meter = meter.to(torch.float64)
        self.assertEqual(meter.y_pred.dtype, torch.float32)
        self.assertEqual(meter.to_numpy().y_pred.dtype, np.float64)
        self.assertAlmostEqual(meter.avg, 1.)
        for name in ['loss', 'mse', 'acc']:
            meter = sw.metrics.get_meter(name, 'torch').to(device='cpu')
            self.assertEqual(meter.device, torch.device('cpu'))
        if torch.backends.mps.is_available():
            meter = sw.metrics.get_meter('auc', 'torch').to('mps')
            meter.update(torch.tensor([0., 1.], device='mps'), torch.tensor([.2, .8], device='mps'))
            self.assertEqual(meter.y_pred.device.type, 'mps')
            self.assertAlmostEqual(meter.avg, 1.)


# hand-computed example, 4 covers and 4 stegos
#   covers .1 .2 .3 .4, stegos .25 .35 .5 .6
#   ROC (FPR, TPR): (0,0) (0,.25) (0,.5) (.25,.5) (.25,.75) (.5,.75) (.5,1) (.75,1) (1,1)
#   P_E = .25, AUC = 13/16 (stego-cover pairs ranked correctly)
#   threshold .3: TP 3, FP 2, TN 2, FN 1
HAND_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
HAND_SCORES = [.1, .2, .3, .4, .25, .35, .5, .6]

DETECTION_KW = {
    'accuracy': [{'threshold': 0.}, {'threshold': .5}],
    'misclassification': [{'threshold': 0.}],
    'precision': [{'threshold': 0.}, {'threshold': 1.}],
    'recall': [{'threshold': 0.}, {'threshold': 1.}],
    'p_e': [{}],
    'roc_auc': [{}],
    'eer': [{}],
    'roc_wauc': [{}, {'implementation': sw.metrics.WAUC_ALASKA}],
    'fnr_at_fpr': [{}, {'strategy': 'below'}, {'strategy': 'above'}, {'fpr_level': .01}, {'fpr_level': .5}],
}


def roc_reference(labels, scores):
    """ROC by scikit-learn, with all points."""
    return skm.roc_curve(labels, scores, pos_label=1, drop_intermediate=False)


class TestMetricsReference(unittest.TestCase):
    """Test suite for metrics, against reference implementations."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_hand_example(self, backend: str):
        self._logger.info(f'TestMetricsReference.test_hand_example({backend=})')
        ys = HAND_LABELS, HAND_SCORES
        self.assertAlmostEqual(compute('pe', backend, *ys), .25)
        self.assertAlmostEqual(compute('auc', backend, *ys), 13/16)
        # FPR = FNR = 1/4 at threshold .35
        self.assertAlmostEqual(compute('eer', backend, *ys), .25)
        self.assertAlmostEqual(compute('acc', backend, *ys, threshold=.3), 5/8)
        self.assertAlmostEqual(compute('misc', backend, *ys, threshold=.3), 3/8)
        self.assertAlmostEqual(compute('precision', backend, *ys, threshold=.3), 3/5)
        self.assertAlmostEqual(compute('recall', backend, *ys, threshold=.3), 3/4)
        for strategy in [None, 'below', 'above']:
            # FPR 10% lies between (0, .5) and (.25, .5)
            self.assertAlmostEqual(compute('fnr_fpr', backend, *ys, fpr_level=.1, strategy=strategy), .5)
        # FPR 37.5% lies between (.25, .75) and (.5, .75), (.5, 1) is above
        self.assertAlmostEqual(compute('fnr_fpr', backend, *ys, fpr_level=.375, strategy='below'), .25)
        meter = sw.metrics.get_meter('pe', backend)
        p_e, _ = meter.given_pe(BACKENDS[backend](HAND_LABELS), BACKENDS[backend](HAND_SCORES), .3)
        self.assertAlmostEqual(p_e, .5 * (2/4 + 1/4))

    @parameterized.expand([[b] for b in BACKENDS])
    def test_pe_threshold(self, backend: str):
        """P_E at a given threshold, e.g., optimized on another set, and the threshold used."""
        self._logger.info(f'TestMetricsReference.test_pe_threshold({backend=})')
        ys = BACKENDS[backend](HAND_LABELS), BACKENDS[backend](HAND_SCORES)
        # optimized
        meter = sw.metrics.get_meter('pe', backend)
        meter.update(*ys)
        self.assertAlmostEqual(meter.avg, .25)
        self.assertAlmostEqual(meter.avg, compute('pe', backend, *ys, threshold=meter.tau))
        # given, as of given_pe
        meter = sw.metrics.get_meter('pe', backend, threshold=.3)
        meter.update(*ys)
        self.assertAlmostEqual(meter.avg, .5 * (2/4 + 1/4))
        self.assertEqual(meter.tau, .3)

    @parameterized.expand([
        (backend, threshold)
        for backend in BACKENDS
        for threshold in [-.5, 0., .5]
    ])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_threshold_metrics(self, backend: str, threshold: float):
        """Accuracy, precision, recall against scikit-learn."""
        self._logger.info(f'TestMetricsReference.test_threshold_metrics({backend=}, {threshold=})')
        labels, scores = scores_and_labels(n=500)
        y_hat = (scores >= threshold).astype(labels.dtype)
        ys = labels, scores
        self.assertAlmostEqual(compute('acc', backend, *ys, threshold=threshold), skm.accuracy_score(labels, y_hat))
        self.assertAlmostEqual(compute('misc', backend, *ys, threshold=threshold), 1 - skm.accuracy_score(labels, y_hat))
        self.assertAlmostEqual(compute('precision', backend, *ys, threshold=threshold), skm.precision_score(labels, y_hat))
        self.assertAlmostEqual(compute('recall', backend, *ys, threshold=threshold), skm.recall_score(labels, y_hat))
        # hard predictions without threshold
        self.assertAlmostEqual(compute('acc', backend, labels, y_hat), skm.accuracy_score(labels, y_hat))

    @parameterized.expand([
        (backend, ties)
        for backend in BACKENDS
        for ties in [False, True]
    ])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_roc_curve(self, backend: str, ties: bool):
        """ROC curve against scikit-learn."""
        self._logger.info(f'TestMetricsReference.test_roc_curve({backend=}, {ties=})')
        labels, scores = scores_and_labels(n=500, ties=ties)
        fpr, tpr, taus = sw.metrics.get_meter('auc', backend).roc(BACKENDS[backend](labels), BACKENDS[backend](scores))
        fpr_ref, tpr_ref, taus_ref = roc_reference(labels, scores)
        np.testing.assert_allclose(np.asarray(fpr), fpr_ref)
        np.testing.assert_allclose(np.asarray(tpr), tpr_ref)
        np.testing.assert_allclose(np.asarray(taus)[1:], taus_ref[1:])

    @parameterized.expand([
        (backend, ties)
        for backend in BACKENDS
        for ties in [False, True]
    ])
    def test_roc_quantile(self, backend: str, ties: bool):
        """Quantile ROC is the exact ROC, sampled at the quantiles."""
        self._logger.info(f'TestMetricsReference.test_roc_quantile({backend=}, {ties=})')
        labels, scores = scores_and_labels(n=500, ties=ties)
        meter = sw.metrics.get_meter('auc', backend)
        fpr, tpr, taus = meter.roc_quantile(BACKENDS[backend](labels), BACKENDS[backend](scores), num_steps=50)
        fpr, tpr, taus = np.asarray(fpr), np.asarray(tpr), np.asarray(taus)
        np.testing.assert_allclose(taus, np.quantile(scores, np.linspace(0, 1, 51)))
        # rates at each threshold, by definition
        np.testing.assert_allclose(fpr, [np.mean(scores[labels == 0] >= tau) for tau in taus], rtol=1e-6)
        np.testing.assert_allclose(tpr, [np.mean(scores[labels == 1] >= tau) for tau in taus], rtol=1e-6)
        # points lie on the exact ROC curve
        fpr_exact, tpr_exact, _ = meter.roc(BACKENDS[backend](labels), BACKENDS[backend](scores))
        exact = np.stack([np.asarray(fpr_exact), np.asarray(tpr_exact)], axis=1)
        for point in np.stack([fpr, tpr], axis=1).astype('float64'):  # quantile ROC is float32
            self.assertLess(np.min(np.abs(exact - point).max(axis=1)), 1e-6)

    @parameterized.expand([
        (backend, level, ties)
        for backend in BACKENDS
        for level in [.01, .05, .3]
        for ties in [False, True]
    ])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_fnr_at_fpr(self, backend: str, level: float, ties: bool):
        """FNR at FPR of all strategies, against the ROC curve of scikit-learn."""
        self._logger.info(f'TestMetricsReference.test_fnr_at_fpr({backend=}, {level=}, {ties=})')
        labels, scores = scores_and_labels(ties=ties)
        fpr, tpr, _ = roc_reference(labels, scores)
        idx_below = np.flatnonzero(fpr <= level)[-1]
        idx_above = idx_below + 1
        alpha = (level - fpr[idx_below]) / (fpr[idx_above] - fpr[idx_below])
        expected = {
            'below': 1 - tpr[idx_below],
            'above': 1 - tpr[idx_above],
            None: 1 - ((1 - alpha) * tpr[idx_below] + alpha * tpr[idx_above]),
        }
        for strategy, fnr in expected.items():
            self.assertAlmostEqual(compute('fnr_fpr', backend, labels, scores, fpr_level=level, strategy=strategy), fnr)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_fnr_at_fpr_edge(self, backend: str):
        self._logger.info(f'TestMetricsReference.test_fnr_at_fpr_edge({backend=})')
        ys = HAND_LABELS, HAND_SCORES
        # all covers falsely detected is allowed
        self.assertAlmostEqual(compute('fnr_fpr', backend, *ys, fpr_level=1.), 0.)
        with self.assertRaises(NotImplementedError):
            compute('fnr_fpr', backend, *ys, strategy='unknown')

    @parameterized.expand([[b] for b in BACKENDS])
    def test_roc_warnings(self, backend: str):
        self._logger.info(f'TestMetricsReference.test_roc_warnings({backend=})')
        roc = sw.metrics.get_meter('auc', backend).roc
        with self.assertWarns(UserWarning):
            roc(BACKENDS[backend]([0., 0.]), BACKENDS[backend]([.1, .2]))
        with self.assertWarns(UserWarning):
            roc(BACKENDS[backend]([1., 1.]), BACKENDS[backend]([.1, .2]))

    @parameterized.expand([[b] for b in BACKENDS])
    def test_undefined_rates(self, backend: str):
        self._logger.info(f'TestMetricsReference.test_undefined_rates({backend=})')
        self.assertTrue(np.isnan(sw.metrics.get_meter('misc', backend).avg))
        self.assertTrue(np.isnan(compute('recall', backend, [0., 0.], [1., 0.])))

    @parameterized.expand([[b] for b in BACKENDS])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_mse_mae(self, backend: str):
        """MSE, MAE, PSNR against scikit-learn, over unequal batches and with mask."""
        self._logger.info(f'TestMetricsReference.test_mse_mae({backend=})')
        rng = np.random.default_rng(12345)
        y_true = rng.uniform(0, 255, size=(10, 8, 8))
        y_pred = y_true + rng.normal(scale=3, size=y_true.shape)
        mask = rng.integers(0, 2, size=y_true.shape)
        for name, reference in [('mse', skm.mean_squared_error), ('mae', skm.mean_absolute_error)]:
            meter = sw.metrics.get_meter(name, backend)
            for i, j in [(0, 7), (7, 10)]:
                meter.update(BACKENDS[backend](y_true[i:j]), BACKENDS[backend](y_pred[i:j]))
            self.assertAlmostEqual(meter.avg, reference(y_true.flatten(), y_pred.flatten()), places=3)
            # masked
            meter = sw.metrics.get_meter(name, backend, mask=1)
            for i, j in [(0, 7), (7, 10)]:
                meter.update(BACKENDS[backend](y_true[i:j]), BACKENDS[backend](y_pred[i:j]), mask=BACKENDS[backend](mask[i:j]))
            self.assertAlmostEqual(meter.avg, reference(y_true[mask == 1], y_pred[mask == 1]), places=3)
            with self.assertRaises(ValueError):
                meter.update(BACKENDS[backend](y_true), BACKENDS[backend](y_pred))
        # PSNR by definition
        meter = sw.metrics.get_meter('psnr', backend)
        meter.update(BACKENDS[backend](y_true), BACKENDS[backend](y_pred))
        expected = 10 * np.log10(255**2 / skm.mean_squared_error(y_true.flatten(), y_pred.flatten()))
        self.assertAlmostEqual(meter.avg, expected, places=3)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_moving_average(self, backend: str):
        """Exponential moving average, computed by hand."""
        self._logger.info(f'TestMetricsReference.test_moving_average({backend=})')
        if backend == 'numpy':
            self.skipTest('numpy meters have no moving average')
        meter = sw.metrics.get_meter('loss', backend, gamma=.5)
        for v in [1., 2.]:
            meter.update(BACKENDS[backend](v))
        self.assertAlmostEqual(meter.avg, .5 * 1 + .5 * 2)
        self.assertAlmostEqual(meter.var, (.5 * 1 + .5 * 4) - 1.5**2)

    def test_ws(self):
        """WS with perfect cover prediction estimates zero change rate."""
        self._logger.info('TestMetricsReference.test_ws()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 256, size=(4, 1, 16, 16)).astype('float64')
        alphas = np.array([0., .1, .2, .4])
        meter = sw.metrics.get_meter('ws', 'numpy')
        meter.update(x, x, alphas)
        self.assertAlmostEqual(meter.avg, np.mean(alphas / 2))


class TestMetricsParity(unittest.TestCase):
    """Test suite for numpy and torch metrics giving the same results."""
    _logger = logging.getLogger(__name__)

    def test_registry(self):
        """Every detection meter exists in both backends, with the same objective."""
        self._logger.info('TestMetricsParity.test_registry()')
        for name in DETECTION_KW:
            cls_numpy = sw.metrics.METER_REGISTRY[(name, 'numpy')]
            cls_torch = sw.metrics.METER_REGISTRY[(name, 'torch')]
            self.assertEqual(cls_numpy.objective, cls_torch.objective, name)
            self.assertEqual(cls_numpy.name_abbr, cls_torch.name_abbr, name)
            self.assertIs(sw.metrics.METER_REGISTRY[(cls_torch.name_abbr, 'torch')], cls_torch)

    @parameterized.expand([
        (name, i, ties)
        for name, kws in DETECTION_KW.items()
        for i in range(len(kws))
        for ties in [False, True]
    ])
    def test_detection(self, name: str, i: int, ties: bool):
        self._logger.info(f'TestMetricsParity.test_detection({name=}, {i=}, {ties=})')
        kw = DETECTION_KW[name][i]
        labels, scores = scores_and_labels(ties=ties)
        meter_numpy = sw.metrics.get_meter(name, 'numpy', **kw)
        meter_torch = sw.metrics.get_meter(name, 'torch', **kw)
        for k in range(0, len(labels), 128):
            meter_numpy.update(labels[k:k+128], scores[k:k+128])
            meter_torch.update(torch.from_numpy(labels[k:k+128]), torch.from_numpy(scores[k:k+128]))
        self.assertAlmostEqual(meter_numpy.avg, meter_torch.avg, places=12)

    def test_torch_inputs(self):
        """Torch meters accept various dtypes and shapes, and gradients."""
        self._logger.info('TestMetricsParity.test_torch_inputs()')
        labels, scores = scores_and_labels(n=200)
        expected = compute('auc', 'numpy', labels, scores)
        variants = [
            (torch.from_numpy(labels).long(), torch.from_numpy(scores).float()),
            (torch.from_numpy(labels).bool(), torch.from_numpy(scores).double()),
            (torch.from_numpy(labels).view(-1, 1), torch.from_numpy(scores).view(-1, 1)),
            (torch.from_numpy(labels), torch.from_numpy(scores).requires_grad_(True)),
        ]
        for y_true, y_pred in variants:
            meter = sw.metrics.get_meter('auc', 'torch')
            meter.update(y_true, y_pred)
            self.assertAlmostEqual(meter.avg, expected, places=6)
        # half precision rounds the scores, compared after rounding
        meter = sw.metrics.get_meter('auc', 'torch')
        meter.update(torch.from_numpy(labels), torch.from_numpy(scores).half())
        self.assertAlmostEqual(meter.avg, compute('auc', 'numpy', labels, torch.from_numpy(scores).half().double().numpy()))

    def test_torch_history(self):
        """Torch meters with history consider only the last samples."""
        self._logger.info('TestMetricsParity.test_torch_history()')
        labels, scores = scores_and_labels()
        rng = np.random.default_rng(12345)
        order = rng.permutation(len(labels))
        labels, scores = labels[order], scores[order]
        meter = sw.metrics.get_meter('auc', 'torch', history=500)
        for k in range(0, len(labels), 128):
            meter.update(torch.from_numpy(labels[k:k+128]), torch.from_numpy(scores[k:k+128]))
        self.assertAlmostEqual(meter.avg, compute('auc', 'numpy', labels[-500:], scores[-500:]), places=12)


# Previous sealwatch implementation of the weighted AUC.

def wauc_previous(y_true, y_pred):
    """Previous sealwatch implementation, wAUCMeter.avg before the switch."""
    metrics = skm
    # ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true,
        y_pred,
        pos_label=1,
        drop_intermediate=False,
    )
    # NaNs in output
    if np.isnan(fpr).any() or np.isnan(tpr).any():
        return np.nan
    # alpha, for which beta=.4
    idx_beta_p4 = np.argmin(tpr < .4)
    alpha_beta_p4 = fpr[idx_beta_p4]
    # split the ROC
    fprA, tprA = fpr[:idx_beta_p4], tpr[:idx_beta_p4]
    fprB, tprB = fpr[idx_beta_p4:], tpr[idx_beta_p4:]
    # integrate
    aucA = metrics.auc(fprA, tprA)
    aucB = metrics.auc(fprB, tprB)
    # sum and weight
    wauc = aucA * 2 + aucB
    # normalize
    wauc = wauc / (1 + alpha_beta_p4)
    return wauc


# Third-party reference implementations, downloaded at a pinned commit, not redistributed.

ALASKA_URL = 'https://raw.githubusercontent.com/NoAchache/Steganalysis-Kaggle-Alaska2-competition-/4125208cf44dfbfcc93bbaf7f47f8a44b1d40045'
DDE_URL = 'https://raw.githubusercontent.com/DDELab/deepsteganalysis/a0ccaa41e4269b90401038252d62f285c921600b'


def import_remote(base_url: str, files: List[str], module: str):
    """Downloads files into a temporary folder, and imports module from it.

    Raises SkipTest if the files cannot be downloaded or imported.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            for f in files:
                os.makedirs(os.path.join(tmpdir, os.path.dirname(f)), exist_ok=True)
                urllib.request.urlretrieve(f'{base_url}/{f}', os.path.join(tmpdir, f))
        except OSError as e:
            raise unittest.SkipTest(f'reference not available: {e}')
        # imported modules and warning filters are restored afterwards
        modules = dict(sys.modules)
        sys.path.insert(0, tmpdir)
        try:
            with warnings.catch_warnings():
                spec = importlib.util.spec_from_file_location(f'reference_{module}', os.path.join(tmpdir, f'{module}.py'))
                reference = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reference)
        except ImportError as e:
            raise unittest.SkipTest(f'reference dependencies not installed: {e}')
        finally:
            sys.path.remove(tmpdir)
            sys.modules.clear()
            sys.modules.update(modules)
    return reference


@functools.lru_cache(maxsize=None)
def alaska_reference():
    """Metric of the ALASKA2 competition, https://www.kaggle.com/anokas/weighted-auc-metric-updated"""
    return import_remote(ALASKA_URL, ['utils/competition_metric.py'], 'utils/competition_metric')


@functools.lru_cache(maxsize=None)
def dde_reference():
    """Metrics of DDELab, https://github.com/DDELab/deepsteganalysis"""
    return import_remote(
        DDE_URL,
        ['metrics/roc_metrics.py', 'tools/decorators.py', 'tools/numpy_utils.py'],
        'metrics/roc_metrics',
    )


def alaska_weighted_auc(y_true, y_valid):
    return alaska_reference().CompetitionMetric.alaska_weighted_auc(y_true, y_valid)


# DDELab metrics return tensors

def dde_wauc(y_true, y_pred):
    return dde_reference().wauc(y_true, y_pred).item()


def dde_md5(y_true, y_pred):
    return dde_reference().md5(y_true, y_pred).item()


def dde_pe(y_true, y_pred):
    return dde_reference().pe(y_true, y_pred).item()


# scenarios without tied scores, where the references are exact
WAUC_SCENARIOS = [
    (backend, n, loc, seed)
    for backend in BACKENDS
    for n in [50, 1000]
    for loc in [.1, .75, 2.]
    for seed in [1, 2]
]


class TestWAUC(unittest.TestCase):
    """Test suite for the weighted AUC, against the reference implementations."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand(WAUC_SCENARIOS)
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_fix_direction_vs_previous(self, backend: str, n: int, loc: float, seed: int):
        """WAUC_FIX_DIRECTION equals the previous implementation."""
        self._logger.info(f'TestWAUC.test_fix_direction_vs_previous({backend=}, {n=}, {loc=}, {seed=})')
        labels, scores = scores_and_labels(n=n, loc=loc, seed=seed)
        wauc = compute('wauc', backend, labels, scores, implementation=sw.metrics.WAUC_FIX_DIRECTION)
        self.assertAlmostEqual(wauc, wauc_previous(labels, scores), places=12)

    @parameterized.expand(WAUC_SCENARIOS)
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_alaska_vs_reference(self, backend: str, n: int, loc: float, seed: int):
        """WAUC_ALASKA equals the metric of the ALASKA2 competition."""
        self._logger.info(f'TestWAUC.test_alaska_vs_reference({backend=}, {n=}, {loc=}, {seed=})')
        labels, scores = scores_and_labels(n=n, loc=loc, seed=seed)
        wauc = compute('wauc', backend, labels, scores, implementation=sw.metrics.WAUC_ALASKA)
        self.assertAlmostEqual(wauc, alaska_weighted_auc(labels, scores), places=12)

    @parameterized.expand([[loc] for loc in [.5, 1.5]])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_ties(self, loc: float):
        """With tied scores, the references miss parts of diagonal ROC segments.

        Both references are then below the exact area, see test_integration.
        """
        self._logger.info(f'TestWAUC.test_ties({loc=})')
        labels, scores = scores_and_labels(loc=loc, ties=True)
        fix = compute('wauc', 'numpy', labels, scores, implementation=sw.metrics.WAUC_FIX_DIRECTION)
        alaska = compute('wauc', 'numpy', labels, scores, implementation=sw.metrics.WAUC_ALASKA)
        self.assertLessEqual(wauc_previous(labels, scores), fix)
        self.assertAlmostEqual(alaska_weighted_auc(labels, scores), alaska, places=2)

    @parameterized.expand([[b] for b in BACKENDS])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_perfect(self, backend: str):
        """Perfect detector, where the references fail."""
        self._logger.info(f'TestWAUC.test_perfect({backend=})')
        ys = [0, 0, 1, 1], [.1, .2, .8, .9]
        for implementation in [sw.metrics.WAUC_FIX_DIRECTION, sw.metrics.WAUC_ALASKA]:
            self.assertAlmostEqual(compute('wauc', backend, *ys, implementation=implementation), 1.)
        # previous fails, integrating a single point
        with self.assertRaises(ValueError):
            wauc_previous(*ys)
        # reference skips the lower band, which the ROC crosses vertically, i.e., (0 + .6) / 1.4
        self.assertAlmostEqual(alaska_weighted_auc(*ys), .6 / 1.4)

    @parameterized.expand([
        (implementation, loc, ties)
        for implementation in ['alaska', 'fix_direction']
        for loc in [.5, 1.5]
        for ties in [False, True]
    ])
    def test_integration(self, implementation: str, loc: float, ties: bool):
        """Both weighted AUCs against numerical integration of the ROC."""
        self._logger.info(f'TestWAUC.test_integration({implementation=}, {loc=}, {ties=})')
        labels, scores = scores_and_labels(loc=loc, ties=ties)
        fpr, tpr, _ = sw.metrics._numpy.wAUCMeter.roc(labels, scores)
        # midpoint rule, the ROC as a function of the FPR
        x = (np.arange(1_000_000) + .5) / 1_000_000
        y = np.interp(x, fpr, tpr)
        if implementation == 'alaska':
            expected = (2 * np.mean(np.clip(y, 0, .4)) + np.mean(np.clip(y, .4, 1) - .4)) / 1.4
            wauc = compute('wauc', 'numpy', labels, scores, implementation=sw.metrics.WAUC_ALASKA)
        else:
            alpha = fpr[np.argmax(tpr >= .4)]
            expected = (2 * np.mean(y * (x < alpha)) + np.mean(y * (x >= alpha))) / (1 + alpha)
            wauc = compute('wauc', 'numpy', labels, scores, implementation=sw.metrics.WAUC_FIX_DIRECTION)
        self.assertAlmostEqual(wauc, expected, places=5)

    @parameterized.expand([[b] for b in BACKENDS])
    def test_implementation(self, backend: str):
        self._logger.info(f'TestWAUC.test_implementation({backend=})')
        labels, scores = scores_and_labels()
        default = compute('wauc', backend, labels, scores)
        fix = compute('wauc', backend, labels, scores, implementation=sw.WAUC_FIX_DIRECTION)
        alaska = compute('wauc', backend, labels, scores, implementation=sw.WAUC_ALASKA)
        self.assertEqual(default, fix)
        self.assertNotAlmostEqual(fix, alaska, places=3)
        with self.assertRaises(NotImplementedError):
            sw.metrics.get_meter('wauc', backend, implementation='unknown')


class TestDDE(unittest.TestCase):
    """Test suite for the metrics, against the reference implementations of DDELab."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand(WAUC_SCENARIOS)
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_vs_dde(self, backend: str, n: int, loc: float, seed: int):
        """wAUC, MD5 (FNR at 5% FPR), and P_E equal the ones of DDELab."""
        self._logger.info(f'TestDDE.test_vs_dde({backend=}, {n=}, {loc=}, {seed=})')
        labels, scores = scores_and_labels(n=n, loc=loc, seed=seed)
        ys = labels, scores
        self.assertAlmostEqual(compute('wauc', backend, *ys, implementation=sw.metrics.WAUC_ALASKA), dde_wauc(*ys), places=12)
        self.assertAlmostEqual(compute('fnr_fpr', backend, *ys, fpr_level=.05), dde_md5(*ys), places=12)
        self.assertAlmostEqual(compute('pe', backend, *ys), dde_pe(*ys), places=12)

    @parameterized.expand([[loc] for loc in [.5, 1.5]])
    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_ties(self, loc: float):
        """With tied scores, MD5 and P_E are still exact, wAUC approximates diagonal segments."""
        self._logger.info(f'TestDDE.test_ties({loc=})')
        labels, scores = scores_and_labels(loc=loc, ties=True)
        ys = labels, scores
        self.assertAlmostEqual(compute('fnr_fpr', 'numpy', *ys, fpr_level=.05), dde_md5(*ys), places=12)
        self.assertAlmostEqual(compute('pe', 'numpy', *ys), dde_pe(*ys), places=12)
        self.assertAlmostEqual(compute('wauc', 'numpy', *ys, implementation=sw.metrics.WAUC_ALASKA), dde_wauc(*ys), places=2)

    @unittest.skipIf(skm is None, 'scikit-learn not installed')
    def test_perfect(self):
        """Perfect detector, where DDELab excludes P_E of zero."""
        self._logger.info('TestDDE.test_perfect()')
        ys = np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])
        self.assertEqual(compute('pe', 'numpy', *ys), 0.)
        # DDELab returns the smallest non-zero P_E on the ROC instead
        self.assertAlmostEqual(dde_pe(*ys), .25)
        self.assertEqual(compute('fnr_fpr', 'numpy', *ys, fpr_level=.05), dde_md5(*ys))
