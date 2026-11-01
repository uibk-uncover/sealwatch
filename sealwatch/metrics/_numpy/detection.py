"""

Author: Benedikt Lorch, Martin Benes
Affiliation: Universitaet Innsbruck
"""

import numpy as np
from typing import Tuple
import warnings

from .common import Meter
from .. import Implementation, register_meter


class DetectionMeter(Meter):
    """Abstract class for detection meter.

    Can be used for metrics that need all the labels and predictions.
    Such metrics are typically derived from the ROC curve or confusion matrix.
    """

    def __init__(self, *args, y_true_idx: int = 0, y_pred_idx: int = 1, threshold: float = None, **kw):
        super().__init__(*args, **kw)
        self.y_true_idx = y_true_idx
        self.y_pred_idx = y_pred_idx
        self.threshold = threshold

    def reset(self):
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, *ys):
        y_true, y_pred = ys[self.y_true_idx], ys[self.y_pred_idx]
        self.y_pred = np.concatenate((self.y_pred, np.ravel(y_pred)))
        self.y_true = np.concatenate((self.y_true, np.ravel(y_true)))

    @property
    def y_hat(self):
        """Hard predictions, thresholded scores if threshold is given."""
        if self.threshold is None:
            return self.y_pred
        return (self.y_pred >= self.threshold).astype(self.y_true.dtype)


@register_meter(name='accuracy', backend='numpy', objective='maximize', name_abbr='acc')
class AccuracyMeter(DetectionMeter):
    @property
    def avg(self):
        if len(self.y_true) == 0:
            return np.nan
        return np.mean(self.y_hat == self.y_true)


@register_meter(name='misclassification', backend='numpy', objective='minimize', name_abbr='misc')
class MisclassificationMeter(DetectionMeter):
    @property
    def avg(self):
        if len(self.y_true) == 0:
            return np.nan
        return np.mean(self.y_hat != self.y_true)


@register_meter(name='precision', backend='numpy', objective='maximize', name_abbr='prc')
class PrecisionMeter(DetectionMeter):
    @property
    def avg(self):
        """Precision, NaN if there are no positive predictions."""
        pp = (self.y_hat == 1).sum()
        if pp == 0:
            return np.nan
        return ((self.y_hat == 1) & (self.y_true == 1)).sum() / pp


@register_meter(name='recall', backend='numpy', objective='maximize', name_abbr='rec')
class RecallMeter(DetectionMeter):
    @property
    def avg(self):
        """Recall, NaN if there are no positive samples."""
        ap = (self.y_true == 1).sum()
        if ap == 0:
            return np.nan
        return ((self.y_hat == 1) & (self.y_true == 1)).sum() / ap


class ROCMeter(DetectionMeter):
    @staticmethod
    def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
        """Area under a piecewise linear curve."""
        return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))

    @staticmethod
    def roc(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact ROC curve, positive is score >= tau.

        One point per distinct score, plus (0, 0) for tau = inf.
        Equivalent to sklearn.metrics.roc_curve(drop_intermediate=False).

        :return: FPR and TPR (non-decreasing), and thresholds (decreasing)
        """
        labels, scores = np.ravel(labels), np.ravel(scores)
        # pre-check class availability
        if (labels == 1).sum() == 0:
            warnings.warn("no positive (stego) samples, TPR undefined")
        if (labels == 0).sum() == 0:
            warnings.warn("no negative (cover) samples, FPR undefined")
        # sort by decreasing score
        order = np.argsort(-scores, kind='stable')
        scores, labels = scores[order], labels[order]
        # last occurrence of each distinct score
        distinct = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
        tps = np.cumsum(labels == 1)[distinct]
        fps = np.cumsum(labels == 0)[distinct]
        #
        with np.errstate(invalid='ignore', divide='ignore'):
            tpr = np.r_[0, tps] / tps[-1]
            fpr = np.r_[0, fps] / fps[-1]
        taus = np.r_[np.inf, scores[distinct]]
        return fpr, tpr, taus

    @staticmethod
    def roc_quantile(labels, scores, num_steps=100):
        """ROC over quantile thresholds.

        Approximates the ROC curve at num_steps+1 thresholds,
        the quantiles of the scores, positive is score >= tau.
        Metrics derived from it are approximate, e.g., P_E is overestimated.

        :return: FPR and TPR (non-increasing), and thresholds (increasing)
        """
        # pre-check class availability
        if (labels == 1).sum() == 0:
            warnings.warn("no positive (stego) samples, TPR undefined")
        if (labels == 0).sum() == 0:
            warnings.warn("no negative (cover) samples, FPR undefined")
        #
        steps = np.linspace(0, 1, num_steps+1, endpoint=True)
        taus = np.quantile(scores, steps)

        # compute tpr, fpr for each tau
        cmp = scores[:, None] >= taus[None, :]  # shape: [N, num_steps]

        is_pos = (labels == 1)[:, None]
        is_neg = (labels == 0)[:, None]

        tpr = (cmp & is_pos).astype('float32').sum(axis=0) / is_pos.astype('float32').sum()
        fpr = (cmp & is_neg).astype('float32').sum(axis=0) / is_neg.astype('float32').sum()

        return fpr, tpr, taus

    @staticmethod
    def _undefined(labels: np.ndarray) -> bool:
        """ROC metrics are undefined unless both classes are present."""
        return len(labels) == 0 or not (labels == 1).any() or not (labels == 0).any()


@register_meter(name='p_e', backend='numpy', objective='minimize', name_abbr='pe')
class PEMeter(ROCMeter):
    """P_E metric, as used in the steganalysis literature.

    P_E = min over tau of (P_FA + P_MD) / 2.

    The threshold tau is optimized on the given data, by default.
    If threshold is given, P_E is evaluated at it instead,
    e.g., at the threshold optimized on the training set, see tau.

    Source: https://github.com/DDELab/deepsteganalysis
    Unlike there, P_E of zero (perfect detector) is not excluded.
    """
    @staticmethod
    def given_pe(labels, scores, tau):
        """P_E for a fixed threshold tau."""
        tpr = np.mean(scores[labels == 1] >= tau)
        fpr = np.mean(scores[labels == 0] >= tau)
        p_e = ((1 - tpr) + fpr) / 2
        return p_e, tau

    @classmethod
    def optimal_pe(cls, labels, scores, **kw):
        """Minimum over the exact ROC curve."""
        fpr, tpr, taus = cls.roc(labels=labels, scores=scores, **kw)

        # P_E
        p_e = ((1 - tpr) + fpr) / 2

        # best index
        idx_best = np.argmin(p_e)
        return p_e[idx_best], taus[idx_best]

    def _pe(self):
        """P_E and its threshold, optimized unless threshold is given."""
        labels, scores = self.y_true, self.y_pred

        # NaNs guard
        if self._undefined(labels) or np.isnan(labels).any() or np.isnan(scores).any():
            return np.nan, np.nan

        if self.threshold is None:
            return self.optimal_pe(labels, scores)
        return self.given_pe(labels, scores, self.threshold)

    @property
    def avg(self):
        return self._pe()[0]

    @property
    def tau(self):
        """Threshold of P_E, optimized unless threshold is given."""
        return self._pe()[1]


@register_meter(name='fnr_at_fpr', backend='numpy', objective='minimize', name_abbr='fnr_fpr')
class FNRatFPRMeter(ROCMeter):
    """False negative rate at k% FPR metric."""
    def __init__(self, *args, fpr_level: float = .05, strategy: str = None, **kw):
        super().__init__(*args, **kw)
        self.fpr_level = fpr_level
        self.strategy = strategy

    @classmethod
    def fnr_at_fpr(cls, labels, scores, fpr_level: float = .05, strategy: str = None, **kw):
        """FNR at the given FPR level.

        :param strategy: None interpolates on the ROC curve,
            'below' takes the best threshold with FPR <= level,
            'above' the first threshold with FPR > level
        :return: FNR and threshold
        """
        fpr, tpr, taus = cls.roc(labels=labels, scores=scores, **kw)
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            return np.nan, np.nan

        # last point with FPR <= level, i.e., the highest TPR
        idx_below = np.flatnonzero(fpr <= fpr_level)[-1]
        if idx_below == len(fpr) - 1:
            return 1 - tpr[idx_below], taus[idx_below]
        idx_above = idx_below + 1

        if strategy is None:
            alpha = (fpr_level - fpr[idx_below])/(fpr[idx_above] - fpr[idx_below])
            tpr_avg = (1-alpha)*tpr[idx_below] + alpha*tpr[idx_above]
            tau_avg = (1-alpha)*taus[idx_below] + alpha*taus[idx_above] if np.isfinite(taus[idx_below]) else taus[idx_above]
            return 1 - tpr_avg, tau_avg

        else:
            if strategy == 'below':
                idx = idx_below
            elif strategy == 'above':
                idx = idx_above
            else:
                raise NotImplementedError(f'unknown strategy {strategy}')
            return 1 - tpr[idx], taus[idx]

    @property
    def avg(self):
        if self._undefined(self.y_true):
            return np.nan
        return self.fnr_at_fpr(
            self.y_true,
            self.y_pred,
            fpr_level=self.fpr_level,
            strategy=self.strategy)[0]


@register_meter(name='eer', backend='numpy', objective='minimize')
class EERMeter(ROCMeter):
    """Equal error rate, at the threshold where the false alarm rate equals the missed detection rate.

    Interpolated linearly on the ROC curve, as fnr_at_fpr.
    The threshold is optimized on the given data, see tau.
    """
    @classmethod
    def eer(cls, labels, scores, **kw):
        """EER, interpolated where FPR crosses FNR on the ROC curve.

        :return: EER and threshold
        """
        fpr, tpr, taus = cls.roc(labels=labels, scores=scores, **kw)
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            return np.nan, np.nan

        # FPR - FNR is non-decreasing, from -1 at (0, 0) to 1 at (1, 1)
        d = fpr - (1 - tpr)
        idx = np.flatnonzero(d >= 0)[0]
        if d[idx] == 0:
            return fpr[idx], taus[idx]

        # interpolate between the points before and after the crossing
        alpha = -d[idx - 1] / (d[idx] - d[idx - 1])
        eer = (1 - alpha) * fpr[idx - 1] + alpha * fpr[idx]
        tau = (1 - alpha) * taus[idx - 1] + alpha * taus[idx] if np.isfinite(taus[idx - 1]) else taus[idx]
        return eer, tau

    @property
    def avg(self):
        if self._undefined(self.y_true):
            return np.nan
        return self.eer(self.y_true, self.y_pred)[0]

    @property
    def tau(self):
        """Threshold of EER."""
        if self._undefined(self.y_true):
            return np.nan
        return self.eer(self.y_true, self.y_pred)[1]


@register_meter(name='roc_auc', backend='numpy', objective='maximize', name_abbr='auc')
class AUCMeter(ROCMeter):

    @classmethod
    def auc(cls, labels, scores, **kw):
        fpr, tpr, taus = cls.roc(labels=labels, scores=scores, **kw)
        return cls._trapezoid(fpr, tpr)

    @property
    def avg(self):
        if self._undefined(self.y_true):
            return np.nan
        return self.auc(self.y_true, self.y_pred)


@register_meter(name='roc_wauc', backend='numpy', objective='maximize', name_abbr='wauc')
class wAUCMeter(ROCMeter):
    """Weighted AUC, weighting the ROC below TPR 0.4 twice.

    Two implementations differ in the direction of the split,
    see :meth:`wauc_fix_direction` and :meth:`wauc_alaska`.
    """
    def __init__(self, *args, implementation: Implementation = Implementation.WAUC_FIX_DIRECTION, **kw):
        super().__init__(*args, **kw)
        if implementation not in (Implementation.WAUC_FIX_DIRECTION, Implementation.WAUC_ALASKA):
            raise NotImplementedError(f'unknown implementation {implementation}')
        self.implementation = implementation

    @classmethod
    def _area_within(cls, fpr: np.ndarray, tpr: np.ndarray, x: Tuple[float, float], y: Tuple[float, float]) -> float:
        """Area under the ROC within the rectangle x times y, above its bottom.

        The points where the ROC crosses the borders of the rectangle are added,
        so that clipping the ROC to the rectangle is exact.
        """
        (x0, x1), (y0, y1) = x, y
        dx, dy = np.diff(fpr), np.diff(tpr)
        # position along the ROC, segment index plus fraction
        pos = [np.arange(len(fpr), dtype='float64')]
        for start, delta, borders in [(fpr[:-1], dx, (x0, x1)), (tpr[:-1], dy, (y0, y1))]:
            for border in borders:
                with np.errstate(divide='ignore', invalid='ignore'):
                    frac = (border - start) / delta
                idx = np.flatnonzero((frac > 0) & (frac < 1))
                pos.append(idx + frac[idx])
        pos = np.sort(np.concatenate(pos))
        idx = np.minimum(pos.astype(int), len(fpr) - 2)
        frac = pos - idx
        xs = fpr[idx] + frac * dx[idx]
        ys = tpr[idx] + frac * dy[idx]
        # clip to the rectangle and integrate
        return cls._trapezoid(np.clip(xs, x0, x1), np.clip(ys, y0, y1) - y0)

    @classmethod
    def wauc_fix_direction(cls, fpr: np.ndarray, tpr: np.ndarray, beta: float = .4) -> float:
        """Weighted AUC, splitting the ROC space vertically.

        Split at FPR alpha, where the TPR reaches beta, into [0, alpha] x [0, 1] and [alpha, 1] x [0, 1].
        The area under the ROC in the first part is weighted twice.
        Normalized by the weighted area of the parts, 2*alpha + (1 - alpha) = 1 + alpha.

        :param fpr: FPR, non-decreasing, from (0, 0) to (1, 1)
        :param tpr: TPR, non-decreasing
        :param beta: TPR, where the split is
        """
        # split vertically, at FPR alpha, where the TPR reaches beta
        splits = [0., fpr[np.argmax(tpr >= beta)], 1.]
        weights = [2, 1]
        wauc = 0.
        for (lo, hi), weight in zip(zip(splits[:-1], splits[1:]), weights):
            # area under the ROC within the vertical strip
            wauc += weight * cls._area_within(fpr, tpr, x=(lo, hi), y=(0., 1.))
        # normalize by the weighted area of the ROC space
        return wauc / np.dot(np.diff(splits), weights)

    @classmethod
    def wauc_alaska(cls, fpr: np.ndarray, tpr: np.ndarray, beta: float = .4) -> float:
        """Weighted AUC of the ALASKA2 competition, splitting the ROC space horizontally.

        Split at TPR beta, into [0, 1] x [0, beta] and [0, 1] x [beta, 1].
        The area under the ROC in the first part is weighted twice.
        Normalized by the weighted area of the parts, 2*beta + (1 - beta) = 1 + beta.

        Equals the reference implementation of the ALASKA2 competition,
        https://www.kaggle.com/anokas/weighted-auc-metric-updated,
        except for ROC segments crossing TPR beta diagonally (tied scores),
        which the reference approximates,
        and ROC crossing a band vertically (e.g., perfect detector),
        which the reference skips.

        :param fpr: FPR, non-decreasing, from (0, 0) to (1, 1)
        :param tpr: TPR, non-decreasing
        :param beta: TPR, where the split is
        """
        # split horizontally, at TPR beta
        splits = [0., beta, 1.]
        weights = [2, 1]
        wauc = 0.
        for (lo, hi), weight in zip(zip(splits[:-1], splits[1:]), weights):
            # area under the ROC within the horizontal band
            wauc += weight * cls._area_within(fpr, tpr, x=(0., 1.), y=(lo, hi))
        # normalize by the weighted area of the ROC space
        return wauc / np.dot(np.diff(splits), weights)

    @property
    def avg(self):
        if self._undefined(self.y_true):
            return np.nan
        # ROC curve
        fpr, tpr, thresholds = self.roc(self.y_true, self.y_pred)
        # NaNs in output
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            return np.nan
        if self.implementation is Implementation.WAUC_ALASKA:
            return self.wauc_alaska(fpr, tpr)
        return self.wauc_fix_direction(fpr, tpr)
