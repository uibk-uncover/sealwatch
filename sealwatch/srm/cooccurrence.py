
from collections import OrderedDict
import enum
import numpy as np
from typing import Dict, Callable

from ..tools import matlab


class Implementation(enum.Enum):
    """MiPOD implementation to choose from."""

    CRM_ORIGINAL = enum.auto()
    """Original MiPOD implementation by Remi Cogranne."""
    CRM_FIX_MIN24 = enum.auto()
    """MiPOD implementation with wet-cost fixes."""


def cooccurrence4(
    D: np.ndarray,
    type: str,
    T: int,
) -> np.ndarray:
    """
    Marginalize to [-T, +T].
    No normalization involved.

    :param D: columns of values from which we want to extract the co-occurrences. Values can be positive, negative, or zero.
    :param T: truncation threshold
    :return: nD co-occurrence table, n-dimensional, where each dimension has (2 * T + 1) values
    """
    # Possible values are [-T, ..., 0, ..., T]
    B = 2 * T + 1
    cooc_mat = np.zeros((B, B, B, B), dtype=int)
    # 4th order co-occurrences
    if type == 'hor':
        L, C, E, R = D[:, :-3], D[:, 1:-2], D[:, 2:-1], D[:, 3:]
    elif type == 'ver':
        L, C, E, R = D[:-3], D[1:-2], D[2:-1], D[3:]
    elif type == 'diag':
        L, C, E, R = D[:-3, :-3], D[1:-2, 1:-2], D[2:-1, 2:-1], D[3:, 3:]
    elif type == 'mdiag':
        L, C, E, R = D[3:, :-3], D[2:-1, 1:-2], D[1:-2, 2:-1], D[:-3, 3:]
    elif type == 'square':
        L, C, E, R = D[1:, :-1], D[1:, 1:], D[:-1, 1:], D[:-1, :-1]
    elif type == 'square-ori':
        Dh, Dv = D[:, :D.shape[0]], D[:, D.shape[0]:]
        L, C, E, D = Dh[1:, :-1], Dv[1:, 1:], Dh[:-1, 1:], Dv[:-1, :-1]
    else:
        raise NotImplementedError(f'type {type} not implemented')

    # A = np.hstack([i.reshape(-1, 1) for i in [L, C, E, R]]).astype('int')
    # f = np.histogramdd(  # 20ms!
    #     A,
    #     bins=[list(range(-T, T+2)) for _ in range(4)],
    #     density=True,
    # )
    # return f[0]

    for i in range(-T, T+1):
        ind = L == i
        C2, E2, R2 = C[ind], E[ind], R[ind]

        for j in range(-T, T+1):
            ind = C2 == j
            E3, R3 = E2[ind], R2[ind]

            for k in range(-T, T+1):
                R4 = R3[E3 == k]

                for l in range(-T, T+1):
                    cooc_mat[T+i, T+j, T+k, T+l] = (R4 == l).sum()

    return cooc_mat / np.sum(cooc_mat)


def cooccurrence3_col(
    D: np.ndarray,
    type: str,
    T: int,
) -> np.ndarray:
    """Marginalize to [-T, +T]. No normalization involved.

    :param D: columns of values from which we want to extract the co-occurrences. Values can be positive, negative, or zero.
    :param T: truncation threshold
    :return: nD co-occurrence table, n-dimensional, where each dimension has (2 * T + 1) values
    """

    # Possible values are [-T, ..., 0, ..., T]
    num_candidates = 2 * T + 1
    cooc_mat = np.zeros([num_candidates] * 3, dtype=int)
    # 3rd order co-occurrences
    # if type == 'hor':
    #     L, C, R = D[:, :-2], D[:, 1:-1], D[:, 2:]
    # elif type == 'ver':
    #     L, C, R = D[:-2], D[1:-1], D[2:]
    # elif type == 'diag':
    #     L, C, R = D[:-2, :-2], D[1:-1, 1:-1], D[2:, 2:]
    # elif type == 'mdiag':
    #     L, C, R = D[2:, :-2], D[1:-1, 1:-1], D[:-2, 2:]
    # elif type == 'col':
    L, C, R = D[:, :, 0], D[:, :, 1], D[:, :, 2]
    # else:
    #     raise NotImplementedError(f'type {type} not implemented')

    #
    # A = L + num_candidates*C + num_candidates**2*R
    for i in range(-T, T+1):
        ind = L == i
        C2, R2 = C[ind], R[ind]

        for j in range(-T, T+1):
            R3 = R2[C2 == j]

            for k in range(-T, T+1):
                cooc_mat[T+i, T+j, T+k] = (R3 == k).sum()
                # cooc_mat[T+i, T+j, T+k, T+l] = ((L==i) & (C==j) & (E==k) & (R==l)).sum()

    return cooc_mat / np.sum(cooc_mat)


def all1st(
    X: np.ndarray,
    q: int,
    *,
    T: int = 2,
    CoocN: Callable = cooccurrence4,
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Co-occurrences of all 1st-order residuals.

    Outputted features:
    1a) spam14h
    1b) spam14v (orthogonal-spam)
    1c) minmax22v
    1d) minmax24
    1e) minmax34v
    1f) minmax41
    1g) minmax34
    1h) minmax48h
    1i) minmax54

    Check Figure 1 in journal HUGO paper.
    """
    #
    g = OrderedDict()

    # 1st-order horizontal/vertical residuals
    # (cropping first and last row to match matlab)
    R = X[1:-1, 2:] - X[1:-1, 1:-1]
    L = X[1:-1, :-2] - X[1:-1, 1:-1]
    U = X[:-2, 1:-1] - X[1:-1, 1:-1]
    D = X[2:, 1:-1] - X[1:-1, 1:-1]
    # quantize
    Rq = np.clip(matlab.round(R / q), -T, T)
    Lq = np.clip(matlab.round(L / q), -T, T)
    Uq = np.clip(matlab.round(U / q), -T, T)
    Dq = np.clip(matlab.round(D / q), -T, T)

    # 1st-order diagonal residuals
    RU = X[:-2, 2:] - X[1:-1, 1:-1]
    LU = X[:-2, :-2] - X[1:-1, 1:-1]
    RD = X[2:, 2:] - X[1:-1, 1:-1]
    LD = X[2:, :-2] - X[1:-1, 1:-1]
    # quantize
    RUq = np.clip(matlab.round(RU / q), -T, T)
    RDq = np.clip(matlab.round(RD / q), -T, T)
    LUq = np.clip(matlab.round(LU / q), -T, T)
    LDq = np.clip(matlab.round(LD / q), -T, T)

    # minmax22
    RLq_min = np.minimum(Rq, Lq)
    UDq_min = np.minimum(Uq, Dq)
    RLq_max = np.maximum(Rq, Lq)
    UDq_max = np.maximum(Uq, Dq)
    g['min22h'] = (CoocN(RLq_min, 'hor', T) + CoocN(UDq_min, 'ver', T))
    # g.min22c = Cooc1(RL_min,3,'col',T) + Cooc1(UD_min,3,'col',T);
    g['max22h'] = (CoocN(RLq_max, 'hor', T) + CoocN(UDq_max, 'ver', T))
    # g.max22c = Cooc1(RL_max,3,'col',T) + Cooc1(UD_max,3,'col',T);
    if directional:
        g['min22v'] = (CoocN(UDq_min, 'hor', T) + CoocN(RLq_min, 'ver', T))
        g['max22v'] = (CoocN(UDq_max, 'hor', T) + CoocN(RLq_max, 'ver', T))

    # spam14h/v
    g['spam14h'] = (CoocN(Rq, 'hor', T) + CoocN(Uq, 'ver', T))
    # g.spam14c = Cooc1(Rq,3,'col',T) + Cooc1(Uq,3,'col',T);
    if directional:
        g['spam14v'] = (CoocN(Uq, 'hor', T) + CoocN(Rq, 'ver', T))

    # minmax24
    RUq_min = np.minimum(Rq, Uq)
    RDq_min = np.minimum(Rq, Dq)
    LUq_min = np.minimum(Lq, Uq)
    LDq_min = np.minimum(Lq, Dq)
    RUq_max = np.maximum(Rq, Uq)
    RDq_max = np.maximum(Rq, Dq)
    LUq_max = np.maximum(Lq, Uq)
    LDq_max = np.maximum(Lq, Dq)
    g['min24'] = (
        + CoocN(np.vstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'hor', T)
        + CoocN(np.hstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'ver', T)
    )
    # Cooc(v[RU_min, RD_min, LU_min, LD_min], 3, 'col', T) + Cooc(h[RU_min, RD_min, LU_min, LD_min], 3, 'col', T)
    g['max24'] = (
        + CoocN(np.vstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'hor', T)
        + CoocN(np.hstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'ver', T)
    )
    # Cooc(v[RU_max, RD_max, LU_max, LD_max], 3, 'col', T) + Cooc(h[RU_max, RD_max), LU_max, LD_max], 3, 'col', T)

    # minmax34
    Uq_min = np.min([Lq, Uq, Rq], axis=0)
    Rq_min = np.min([Uq, Rq, Dq], axis=0)
    Dq_min = np.min([Rq, Dq, Lq], axis=0)
    Lq_min = np.min([Dq, Lq, Uq], axis=0)
    Uq_max = np.max([Lq, Uq, Rq], axis=0)
    Rq_max = np.max([Uq, Rq, Dq], axis=0)
    Dq_max = np.max([Rq, Dq, Lq], axis=0)
    Lq_max = np.max([Dq, Lq, Uq], axis=0)
    g['min34h'] = (
        + CoocN(np.vstack([Uq_min, Dq_min]), 'hor', T)
        + CoocN(np.hstack([Lq_min, Rq_min]), 'ver', T)
    )
    # Cooc(h[Uq_min, Dq_min], 3, 'col', T) + Cooc(v[Rq_min, Lq_min], 3, 'col', T)
    g['max34h'] = (
        + CoocN(np.vstack([Uq_max, Dq_max]), 'hor', T)
        + CoocN(np.hstack([Rq_max, Lq_max]), 'ver', T)
    )
    # Cooc(h[Uq_max, Dq_max], 3, 'col', T) + Cooc1(v[Rq_max, Lq_max], 3, 'col', T)
    if directional:
        g['min34v'] = (
            + CoocN(np.hstack([Uq_min, Dq_min]), 'ver', T)
            + CoocN(np.vstack([Rq_min, Lq_min]), 'hor', T)
        )
        g['max34v'] = (
            + CoocN(np.hstack([Uq_max, Dq_max]), 'ver', T)
            + CoocN(np.vstack([Rq_max, Lq_max]), 'hor', T)
        )

    # minmax41
    R_min = np.minimum(RLq_min, UDq_min)
    R_max = np.maximum(RLq_max, UDq_max)
    g['min41'] = CoocN(R_min, 'hor', T)  # (CoocN(R_min, 'hor', T) + CoocN(R_min, 'ver', T))
    # Cooc(R_min, 3, 'col', T)
    g['max41'] = CoocN(R_max, 'hor', T)  # (CoocN(R_max, 'hor', T) + CoocN(R_max, 'ver', T))
    # Cooc(R_max, 3, 'col', T)
    if directional:
        g['min41'] = g['min41'] + CoocN(R_min, 'ver', T)
        g['max41'] = g['max41'] + CoocN(R_max, 'ver', T)

    # minmax34
    RUq_min = np.minimum(RUq_min, RUq)
    RDq_min = np.minimum(RDq_min, RDq)
    LUq_min = np.minimum(LUq_min, LUq)
    LDq_min = np.minimum(LDq_min, LDq)
    RUq_max = np.maximum(RUq_max, RUq)
    RDq_max = np.maximum(RDq_max, RDq)
    LUq_max = np.maximum(LUq_max, LUq)
    LDq_max = np.maximum(LDq_max, LDq)
    g['min34'] = (
        + CoocN(np.vstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'hor', T)
        + CoocN(np.hstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'ver', T)
    )
    # Cooc(v[RU_min, RD_min, LU_min, LD_min], 3, 'col', T) + Cooc(h[RU_min, RD_min, LU_min, LD_min], 3, 'col', T)
    g['max34'] = (
        + CoocN(np.vstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'hor', T)
        + CoocN(np.hstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'ver', T)
    )
    # Cooc(v[RU_max, RD_max, LU_max, LD_max], 3, 'col', T) + Cooc(h[RU_max, RD_max, LU_max, LD_max], 3, 'col', T)

    # minmax48h - outputting both but Figure 1 in our paper lists only 48h.
    RUq_min2 = np.minimum(RUq_min, LUq)
    RDq_min2 = np.minimum(RDq_min, RUq)
    LDq_min2 = np.minimum(LDq_min, RDq)
    LUq_min2 = np.minimum(LUq_min, LDq)
    RUq_min3 = np.minimum(RUq_min, RDq)
    RDq_min3 = np.minimum(RDq_min, LDq)
    LDq_min3 = np.minimum(LDq_min, LUq)
    LUq_min3 = np.minimum(LUq_min, RUq)
    g['min48h'] = (
        + CoocN(np.vstack([RUq_min2, LDq_min2, RDq_min3, LUq_min3]), 'hor', T)
        + CoocN(np.hstack([RDq_min2, LUq_min2, RUq_min3, LDq_min3]), 'ver', T)
    )
    # Cooc(v[RU_min2, LD_min2, RD_min3, LU_min3], 3, 'col', T) + Cooc1(h[RD_min2, LU_min2, RU_min3, LD_min3], 3, 'col', T)
    if directional:
        g['min48v'] = (
            + CoocN(np.vstack([RDq_min2, LUq_min2, RUq_min3, LDq_min3]), 'hor', T)
            + CoocN(np.hstack([RUq_min2, LDq_min2, RDq_min3, LUq_min3]), 'ver', T)
        )

    RUq_max2 = np.maximum(RUq_max, LUq)
    RDq_max2 = np.maximum(RDq_max, RUq)
    LDq_max2 = np.maximum(LDq_max, RDq)
    LUq_max2 = np.maximum(LUq_max, LDq)
    RUq_max3 = np.maximum(RUq_max, RDq)
    RDq_max3 = np.maximum(RDq_max, LDq)
    LDq_max3 = np.maximum(LDq_max, LUq)
    LUq_max3 = np.maximum(LUq_max, RUq)
    g['max48h'] = (
        + CoocN(np.vstack([RUq_max2, LDq_max2, RDq_max3, LUq_max3]), 'hor', T)
        + CoocN(np.hstack([RDq_max2, LUq_max2, RUq_max3, LDq_max3]), 'ver', T)
    )
    # Cooc(v[RU_max2,LD_max2,RD_max3,LU_max3],3,'col',T) + Cooc(h[RD_max2,LU_max2,RU_max3,LD_max3],3,'col',T)
    if directional:
        g['max48v'] = (
            + CoocN(np.vstack([RDq_max2, LUq_max2, RUq_max3, LDq_max3]), 'hor', T)
            + CoocN(np.hstack([RUq_max2, LDq_max2, RDq_max3, LUq_max3]), 'ver', T)
        )

    # minmax54
    RUq_min4 = np.minimum(RUq_min2, RDq)
    RDq_min4 = np.minimum(RDq_min2, LDq)
    LDq_min4 = np.minimum(LDq_min2, LUq)
    LUq_min4 = np.minimum(LUq_min2, RUq)
    RUq_min5 = np.minimum(RUq_min3, LUq)
    RDq_min5 = np.minimum(RDq_min3, RUq)
    LDq_min5 = np.minimum(LDq_min3, RDq)
    LUq_min5 = np.minimum(LUq_min3, LDq)
    g['min54'] = (
        + CoocN(np.vstack([RUq_min4, LDq_min4, RDq_min5, LUq_min5]), 'hor', T)
        + CoocN(np.hstack([RDq_min4, LUq_min4, RUq_min5, LDq_min5]), 'ver', T)
    )
    # Cooc(v[RU_min4,LD_min4,RD_min5,LU_min5],3,'col',T) + Cooc(h[RD_min4,LU_min4,RU_min5,LD_min5],3,'col',T)
    RUq_max4 = np.maximum(RUq_max2, RDq)
    RDq_max4 = np.maximum(RDq_max2, LDq)
    LDq_max4 = np.maximum(LDq_max2, LUq)
    LUq_max4 = np.maximum(LUq_max2, RUq)
    RUq_max5 = np.maximum(RUq_max3, LUq)
    RDq_max5 = np.maximum(RDq_max3, RUq)
    LDq_max5 = np.maximum(LDq_max3, RDq)
    LUq_max5 = np.maximum(LUq_max3, LDq)
    g['max54'] = (
        + CoocN(np.vstack([RUq_max4, LDq_max4, RDq_max5, LUq_max5]), 'hor', T)
        + CoocN(np.hstack([RDq_max4, LUq_max4, RUq_max5, LDq_max5]), 'ver', T)
    )
    #
    return g


def all2nd(
    X: np.ndarray,
    q: int,
    T: int = 2,
    CoocN: Callable = cooccurrence4,
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Co-occurrences of all 2nd-order residuals.

    Outputted features:
    1a) spam12h
    1b) spam12v (orthogonal-spam)
    1c) minmax21
    1d) minmax41
    1e) minmax24h (24v is also outputted but not listed in Figure 1)
    1f) minmax32
    """
    #
    g = OrderedDict()

    # 2nd-order horizontal/vertical residuals
    # (cropping first and last row to match matlab)
    Dh = X[1:-1, :-2] - 2*X[1:-1, 1:-1] + X[1:-1, 2:]
    Dv = X[:-2, 1:-1] - 2*X[1:-1, 1:-1] + X[2:, 1:-1]
    Dd = X[:-2, :-2] - 2*X[1:-1, 1:-1] + X[2:, 2:]
    Dm = X[:-2, 2:] - 2*X[1:-1, 1:-1] + X[2:, :-2]

    # Quantize
    Yh = np.clip(matlab.round(Dh / q), -T, T)
    Yv = np.clip(matlab.round(Dv / q), -T, T)
    Yd = np.clip(matlab.round(Dd / q), -T, T)
    Ym = np.clip(matlab.round(Dm / q), -T, T)

    # spam12h/v
    g['spam12h'] = (CoocN(Yh, 'hor', T) + CoocN(Yv, 'ver', T))
    if directional:
        g['spam12v'] = (CoocN(Yh, 'ver', T) + CoocN(Yv, 'hor', T))

    # minmax21
    Dmin = np.minimum(Yh, Yv)
    Dmax = np.maximum(Yh, Yv)
    g['min21'] = CoocN(Dmin, 'hor', T)
    g['max21'] = CoocN(Dmax, 'hor', T)
    if directional:
        g['min21'] += CoocN(Dmin, 'ver', T)
        g['max21'] += CoocN(Dmax, 'ver', T)

    # minmax41
    Dmin2 = np.min([Dmin, Yd, Ym], axis=0)
    Dmax2 = np.max([Dmax, Yd, Ym], axis=0)
    g['min41'] = CoocN(Dmin2, 'hor', T)
    g['max41'] = CoocN(Dmax2, 'hor', T)
    if directional:
        g['min41'] += CoocN(Dmin2, 'ver', T)
        g['max41'] += CoocN(Dmax2, 'ver', T)

    # minmax32
    RUq_min = np.minimum(Dmin, Ym)
    RDq_min = np.minimum(Dmin, Yd)
    RUq_max = np.maximum(Dmax, Ym)
    RDq_max = np.maximum(Dmax, Yd)
    g['min32'] = CoocN(np.vstack([RUq_min, RDq_min]), 'hor', T)
    g['max32'] = CoocN(np.vstack([RUq_max, RDq_max]), 'hor', T)
    if directional:
        g['min32'] += CoocN(np.hstack([RUq_min, RDq_min]), 'ver', T)
        g['max32'] += CoocN(np.hstack([RUq_max, RDq_max]), 'ver', T)

    # minmax24h
    RUq_min2 = np.minimum(Ym, Yh)
    RDq_min2 = np.minimum(Yd, Yh)
    RUq_min3 = np.minimum(Ym, Yv)
    LUq_min3 = np.minimum(Yd, Yv)
    g['min24h'] = (
        + CoocN(np.vstack([RUq_min2, RDq_min2]), 'hor', T)
        + CoocN(np.hstack([RUq_min3, LUq_min3]), 'ver', T)
    )
    if directional:
        g['min24v'] = (
            + CoocN(np.vstack([RUq_min3, LUq_min3]), 'hor', T)
            + CoocN(np.hstack([RUq_min2, RDq_min2]), 'ver', T)
        )
    RUq_max2 = np.maximum(Ym, Yh)
    RDq_max2 = np.maximum(Yd, Yh)
    RUq_max3 = np.maximum(Ym, Yv)
    LUq_max3 = np.maximum(Yd, Yv)
    g['max24h'] = (
        + CoocN(np.vstack([RUq_max2, RDq_max2]), 'hor', T)
        + CoocN(np.hstack([RUq_max3, LUq_max3]), 'ver', T)
    )
    if directional:
        g['max24v'] = (
            + CoocN(np.vstack([RUq_max3, LUq_max3]), 'hor', T)
            + CoocN(np.hstack([RUq_max2, RDq_max2]), 'ver', T)
        )
    #
    return g


def all3rd(
    X: np.ndarray,
    q: int,
    *,
    T: int = 2,
    CoocN: Callable = cooccurrence4,
    directional: bool = True,
    implementation: Implementation = Implementation.CRM_ORIGINAL,
) -> Dict[str, np.ndarray]:
    """Co-occurrences of all 3rd-order residuals.

    Outputted features:
    1a) spam14h
    1b) spam14v (orthogonal-spam)
    1c) minmax22v
    1d) minmax24
    1e) minmax34v
    1f) minmax41
    1g) minmax34
    1h) minmax48h
    1i) minmax54
    """
    g = OrderedDict()

    # 3rd-order horizontal/vertical residuals
    # R = get_noise_residual(X[:, 1:], 3, 'hor')
    # L = -get_noise_residual(X[:, :-1], 3, 'hor')
    # U = -get_noise_residual(X[:-1, :], 3, 'ver')
    # D = get_noise_residual(X[1:, :], 3, 'ver')
    R = -X[2:-2, 4:] + 3*X[2:-2, 3:-1] - 3*X[2:-2, 2:-2] + X[2:-2, 1:-3]
    L = -X[2:-2, :-4] + 3*X[2:-2, 1:-3] - 3*X[2:-2, 2:-2] + X[2:-2, 3:-1]
    U = -X[:-4, 2:-2] + 3*X[1:-3, 2:-2] - 3*X[2:-2, 2:-2] + X[3:-1, 2:-2]
    D = -X[4:, 2:-2] + 3*X[3:-1, 2:-2] - 3*X[2:-2, 2:-2] + X[1:-3, 2:-2]
    # quantize
    Rq = np.clip(matlab.round(R / q), -T, T)
    Lq = np.clip(matlab.round(L / q), -T, T)
    Uq = np.clip(matlab.round(U / q), -T, T)
    Dq = np.clip(matlab.round(D / q), -T, T)

    # 3rd-order diagonal residuals
    # RU = get_noise_residual(X[:-1, 1:], 3, 'mdiag')
    # LU = -get_noise_residual(X[:-1, :-1], 3, 'diag')
    # RD = get_noise_residual(X[1:, 1:], 3, 'diag')
    # LD = -get_noise_residual(X[1:, :-1], 3, 'mdiag')
    RU = -X[:-4, 4:] + 3*X[1:-3, 3:-1] - 3*X[2:-2, 2:-2] + X[3:-1, 1:-3]
    LU = -X[:-4, :-4] + 3*X[1:-3, 1:-3] - 3*X[2:-2, 2:-2] + X[3:-1, 3:-1]
    RD = -X[4:, 4:] + 3*X[3:-1, 3:-1] - 3*X[2:-2, 2:-2] + X[1:-3, 1:-3]
    LD = -X[4:, :-4] + 3*X[3:-1, 1:-3] - 3*X[2:-2, 2:-2] + X[1:-3, 3:-1]
    # quantize
    RUq = np.clip(matlab.round(RU / q), -T, T)
    LUq = np.clip(matlab.round(LU / q), -T, T)
    RDq = np.clip(matlab.round(RD / q), -T, T)
    LDq = np.clip(matlab.round(LD / q), -T, T)

    # minmax22
    RLq_min = np.minimum(Rq, Lq)
    UDq_min = np.minimum(Uq, Dq)
    RLq_max = np.maximum(Rq, Lq)
    UDq_max = np.maximum(Uq, Dq)
    g['min22h'] = CoocN(RLq_min, 'hor', T) + CoocN(UDq_min, 'ver', T)
    g['max22h'] = CoocN(RLq_max, 'hor', T) + CoocN(UDq_max, 'ver', T)
    if directional:
        g['min22v'] = (CoocN(RLq_min, 'ver', T) + CoocN(UDq_min, 'hor', T))
        g['max22v'] = (CoocN(RLq_max, 'ver', T) + CoocN(UDq_max, 'hor', T))

    # minmax34h
    Uq_min = np.min([RLq_min, Uq], axis=0)
    Rq_min = np.min([UDq_min, Rq], axis=0)
    Dq_min = np.min([RLq_min, Dq], axis=0)
    Lq_min = np.min([UDq_min, Lq], axis=0)
    Uq_max = np.max([RLq_max, Uq], axis=0)
    Rq_max = np.max([UDq_max, Rq], axis=0)
    Dq_max = np.max([RLq_max, Dq], axis=0)
    Lq_max = np.max([UDq_max, Lq], axis=0)
    g['min34h'] = (
        + CoocN(np.vstack([Uq_min, Dq_min]), 'hor', T)
        + CoocN(np.hstack([Rq_min, Lq_min]), 'ver', T)
    )
    g['max34h'] = (
        + CoocN(np.vstack([Uq_max, Dq_max]), 'hor', T)
        + CoocN(np.hstack([Rq_max, Lq_max]), 'ver', T)
    )
    if directional:
        g['min34v'] = (
            + CoocN(np.vstack([Rq_min, Lq_min]), 'hor', T)
            + CoocN(np.hstack([Uq_min, Dq_min]), 'ver', T)
        )
        g['max34v'] = (
            + CoocN(np.vstack([Rq_max, Lq_max]), 'hor', T)
            + CoocN(np.hstack([Uq_max, Dq_max]), 'ver', T)
        )

    # spam14
    g['spam14h'] = (CoocN(Rq, 'hor', T) + CoocN(Uq, 'ver', T))
    if directional:
        g['spam14v'] = (CoocN(Rq, 'ver', T) + CoocN(Uq, 'hor', T))

    # minmax24
    RUq_min = np.minimum(Rq, Uq)
    RDq_min = np.minimum(Rq, Dq)
    LUq_min = np.minimum(Lq, Uq)
    LDq_min = np.minimum(Lq, Dq)
    RUq_max = np.maximum(Rq, Uq)
    RDq_max = np.maximum(Rq, Dq)
    LUq_max = np.maximum(Lq, Uq)
    LDq_max = np.maximum(Lq, Dq)
    g['min24'] = CoocN(np.vstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'hor', T)
    g['max24'] = CoocN(np.vstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'hor', T)
    if implementation == Implementation.CRM_ORIGINAL or directional:  # directional:  # TODO: mistake in DDE's CRM
        g['min24'] += CoocN(np.hstack([RUq_min, RDq_min, LUq_min, LDq_min]), 'ver', T)
        g['max24'] += CoocN(np.hstack([RUq_max, RDq_max, LUq_max, LDq_max]), 'ver', T)

    # minmax41
    R_min = np.minimum(RUq_min, LDq_min)
    R_max = np.maximum(RUq_max, LDq_max)
    g['min41'] = CoocN(R_min, 'hor', T)
    g['max41'] = CoocN(R_max, 'hor', T)
    if directional:
        g['min41'] += CoocN(R_min, 'ver', T)
        g['max41'] += CoocN(R_max, 'ver', T)

    # minmax34
    RUq_min2 = np.minimum(RUq_min, RUq)
    RDq_min2 = np.minimum(RDq_min, RDq)
    LUq_min2 = np.minimum(LUq_min, LUq)
    LDq_min2 = np.minimum(LDq_min, LDq)
    RUq_max2 = np.maximum(RUq_max, RUq)
    RDq_max2 = np.maximum(RDq_max, RDq)
    LUq_max2 = np.maximum(LUq_max, LUq)
    LDq_max2 = np.maximum(LDq_max, LDq)
    g['min34'] = CoocN(np.vstack([RUq_min2, RDq_min2, LUq_min2, LDq_min2]), 'hor', T)
    g['max34'] = CoocN(np.vstack([RUq_max2, RDq_max2, LUq_max2, LDq_max2]), 'hor', T)
    if directional:
        g['min34'] += CoocN(np.hstack([RUq_min2, RDq_min2, LUq_min2, LDq_min2]), 'ver', T)
        g['max34'] += CoocN(np.hstack([RUq_max2, RDq_max2, LUq_max2, LDq_max2]), 'ver', T)

    # minmax48h
    # (outputting both but Figure 1 in our paper lists only 48h)
    RUq_min3 = np.minimum(RUq_min2, LUq)
    RDq_min3 = np.minimum(RDq_min2, RUq)
    LDq_min3 = np.minimum(LDq_min2, RDq)
    LUq_min3 = np.minimum(LUq_min2, LDq)
    RUq_min4 = np.minimum(RUq_min2, RDq)
    RDq_min4 = np.minimum(RDq_min2, LDq)
    LDq_min4 = np.minimum(LDq_min2, LUq)
    LUq_min4 = np.minimum(LUq_min2, RUq)
    g['min48h'] = (
        + CoocN(np.vstack([RUq_min3, LDq_min3, RDq_min4, LUq_min4]), 'hor', T)
        + CoocN(np.hstack([RDq_min3, LUq_min3, RUq_min4, LDq_min4]), 'ver', T)
    )
    if directional:
        g['min48v'] = (
            + CoocN(np.hstack([RUq_min3, LDq_min3, RDq_min4, LUq_min4]), 'ver', T)
            + CoocN(np.vstack([RDq_min3, LUq_min3, RUq_min4, LDq_min4]), 'hor', T)
        )
    RUq_max3 = np.maximum(RUq_max2, LUq)
    RDq_max3 = np.maximum(RDq_max2, RUq)
    LDq_max3 = np.maximum(LDq_max2, RDq)
    LUq_max3 = np.maximum(LUq_max2, LDq)
    RUq_max4 = np.maximum(RUq_max2, RDq)
    RDq_max4 = np.maximum(RDq_max2, LDq)
    LDq_max4 = np.maximum(LDq_max2, LUq)
    LUq_max4 = np.maximum(LUq_max2, RUq)
    g['max48h'] = (
        + CoocN(np.vstack([RUq_max3, LDq_max3, RDq_max4, LUq_max4]), 'hor', T)
        + CoocN(np.hstack([RDq_max3, LUq_max3, RUq_max4, LDq_max4]), 'ver', T)
    )
    if directional:
        g['max48v'] = (
            + CoocN(np.vstack([RDq_max3, LUq_max3, RUq_max4, LDq_max4]), 'hor', T)
            + CoocN(np.hstack([RUq_max3, LDq_max3, RDq_max4, LUq_max4]), 'ver', T)
        )

    # minmax54 -- to be symmetrized as mnmx, directional, hv-symmetrical
    RUq_min5 = np.minimum(RUq_min3, RDq)
    RDq_min5 = np.minimum(RDq_min3, LDq)
    LDq_min5 = np.minimum(LDq_min3, LUq)
    LUq_min5 = np.minimum(LUq_min3, RUq)
    RUq_max5 = np.maximum(RUq_max3, RDq)
    RDq_max5 = np.maximum(RDq_max3, LDq)
    LDq_max5 = np.maximum(LDq_max3, LUq)
    LUq_max5 = np.maximum(LUq_max3, RUq)
    g['min54'] = CoocN(np.vstack([RUq_min5, LDq_min5, RDq_min5, LUq_min5]), 'hor', T)
    g['max54'] = CoocN(np.vstack([RUq_max5, LDq_max5, RDq_max5, LUq_max5]), 'hor', T)
    if directional:
        g['min54'] += CoocN(np.hstack([RDq_min5, LUq_min5, RUq_min5, LDq_min5]), 'ver', T)
        g['max54'] += CoocN(np.hstack([RDq_max5, LUq_max5, RUq_max5, LDq_max5]), 'ver', T)

    #
    return g


def all3x3(
    X: np.ndarray,
    q: int,
    T: int = 2,
    CoocN: Callable = cooccurrence4,
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Co-occurrences of residuals based on KB kernel and its halfes (EDGE residuals)."""
    g = OrderedDict()

    # KB residual
    # D = get_noise_residual(X, 2, 'KB')
    D = (
        -X[:-2, :-2] + 2*X[1:-1, :-2] - X[2:, :-2]
        + 2*X[:-2, 1:-1] - 4*X[1:-1, 1:-1] + 2*X[2:, 1:-1]
        - X[:-2, 2:] + 2*X[1:-1, 2:] - X[2:, 2:]
    )
    # quantize
    Y = np.clip(matlab.round(D / q), -T, T)

    # spam11
    g['spam11'] = CoocN(Y, 'hor', T)
    if directional:
        g['spam11'] += CoocN(Y, 'ver', T)

    # EDGE residuals
    # D = get_noise_residual(X, 2, 'edge-h')
    # Du = D[:,:D.shape[1]//2]
    # Db = D[:,D.shape[1]//2:]
    # D = get_noise_residual(X, 2, 'edge-v')
    # Dl = D[:,:D.shape[1]//2]
    # Dr = D[:,D.shape[1]//2:]
    Du = 2*X[:-2, 1:-1] + 2*X[1:-1, :-2] + 2*X[1:-1, 2:] - X[:-2, :-2] - X[:-2, 2:] - 4*X[1:-1, 1:-1]
    Db = 2*X[2:, 1:-1] + 2*X[1:-1, :-2] + 2*X[1:-1, 2:] - X[2:, :-2] - X[2:, 2:] - 4*X[1:-1, 1:-1]
    Dl = 2*X[1:-1, :-2] + 2*X[:-2, 1:-1] + 2*X[2:, 1:-1] - X[:-2, :-2] - X[2:, :-2] - 4*X[1:-1, 1:-1]
    Dr = 2*X[1:-1, 2:] + 2*X[:-2, 1:-1] + 2*X[2:, 1:-1] - X[:-2, 2:] - X[2:, 2:] - 4*X[1:-1, 1:-1]
    # quantize
    Yu = np.clip(matlab.round(Du / q), -T, T)
    Yb = np.clip(matlab.round(Db / q), -T, T)
    Yl = np.clip(matlab.round(Dl / q), -T, T)
    Yr = np.clip(matlab.round(Dr / q), -T, T)

    # spam14
    g['spam14h'] = (
        + CoocN(np.vstack([Yu, Yb]), 'hor', T)
        + CoocN(np.hstack([Yl, Yr]), 'ver', T)
    )
    if directional:
        g['spam14v'] = (
            + CoocN(np.hstack([Yu, Yb]), 'ver', T)
            + CoocN(np.vstack([Yl, Yr]), 'hor', T)
        )

    # minmax24
    Dmin1 = np.minimum(Yu, Yl)
    Dmin2 = np.minimum(Yb, Yr)
    Dmin3 = np.minimum(Yu, Yr)
    Dmin4 = np.minimum(Yb, Yl)
    Dmax1 = np.maximum(Yu, Yl)
    Dmax2 = np.maximum(Yb, Yr)
    Dmax3 = np.maximum(Yu, Yr)
    Dmax4 = np.maximum(Yb, Yl)
    g['min24'] = CoocN(np.vstack([Dmin1, Dmin2, Dmin3, Dmin4]), 'hor', T)
    g['max24'] = CoocN(np.vstack([Dmax1, Dmax2, Dmax3, Dmax4]), 'hor', T)
    if directional:
        g['min24'] += CoocN(np.hstack([Dmin1, Dmin2, Dmin3, Dmin4]), 'ver', T)
        g['max24'] += CoocN(np.hstack([Dmax1, Dmax2, Dmax3, Dmax4]), 'ver', T)

    # minmax22
    UEq_min = np.minimum(Yu, Yb)
    REq_min = np.minimum(Yr, Yl)
    UEq_max = np.maximum(Yu, Yb)
    REq_max = np.maximum(Yr, Yl)
    g['min22h'] = (CoocN(UEq_min, 'hor', T) + CoocN(REq_min, 'ver', T))
    g['max22h'] = (CoocN(UEq_max, 'hor', T) + CoocN(REq_max, 'ver', T))
    if directional:
        g['min22v'] = (CoocN(UEq_min, 'ver', T) + CoocN(REq_min, 'hor', T))
        g['max22v'] = (CoocN(UEq_max, 'ver', T) + CoocN(REq_max, 'hor', T))

    # minmax41
    Dmin5 = np.minimum(Dmin1, Dmin2)
    Dmax5 = np.maximum(Dmax1, Dmax2)
    g['min41'] = CoocN(Dmin5, 'hor', T)
    g['max41'] = CoocN(Dmax5, 'hor', T)
    if directional:
        g['min41'] += CoocN(Dmin5, 'ver', T)
        g['max41'] += CoocN(Dmax5, 'ver', T)

    #
    return g


def all5x5(
    X: np.ndarray,
    q: int,
    T: int = 2,
    CoocN: Callable = cooccurrence4,
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Co-occurrences of residuals based on KV kernel and its halfes (EDGE residuals)."""
    #
    g = OrderedDict()

    # KV coocurrences
    # D = get_noise_residual(X, 3, 'KV')
    D = (
        + 8*X[1:-3, 2:-2]+8*X[3:-1, 2:-2]+8*X[2:-2, 1:-3]+8*X[2:-2, 3:-1]
        - 6*X[1:-3, 3:-1]-6*X[1:-3, 1:-3]-6*X[3:-1, 1:-3]-6*X[3:-1, 3:-1]
        - 2*X[:-4, 2:-2]-2*X[4:, 2:-2]-2*X[2:-2, 4:]-2*X[2:-2, :-4]
        + 2*X[1:-3, :-4]+2*X[:-4, 1:-3]+2*X[:-4, 3:-1]+2*X[1:-3, 4:]
        + 2*X[3:-1, 4:]+2*X[4:, 3:-1]+2*X[4:, 1:-3]+2*X[3:-1, :-4]
        - X[:-4, :-4]-X[:-4, 4:]-X[4:, :-4]-X[4:, 4:]-12*X[2:-2, 2:-2])
    # quantize
    Y = np.clip(matlab.round(D / q), -T, T)

    # spam11
    g['spam11'] = CoocN(Y, 'hor', T)
    if directional:
        g['spam11'] += CoocN(Y, 'ver', T)

    # EDGE residuals
    Du = (
        + 8*X[2:-2, 1:-3]+8*X[1:-3, 2:-2]+8*X[2:-2, 3:-1]
        - 6*X[1:-3, 1:-3]-6*X[1:-3, 3:-1]
        - 2*X[2:-2, :-4]-2*X[2:-2, 4:]-2*X[:-4, 2:-2]
        + 2*X[1:-3, :-4]+2*X[:-4, 1:-3]+2*X[:-4, 3:-1]+2*X[1:-3, 4:]
        - X[:-4, :-4]-X[:-4, 4:]-12*X[2:-2, 2:-2])
    Dr = (
        + 8*X[1:-3, 2:-2]+8*X[2:-2, 3:-1]+8*X[3:-1, 2:-2]
        - 6*X[1:-3, 3:-1]-6*X[3:-1, 3:-1]
        - 2*X[:-4, 2:-2]-2*X[4:, 2:-2]-2*X[2:-2, 4:]
        + 2*X[:-4, 3:-1]+2*X[1:-3, 4:]+2*X[3:-1, 4:]+2*X[4:, 3:-1]
        - X[:-4, 4:]-X[4:, 4:]-12*X[2:-2, 2:-2])
    Db = (
        + 8*X[2:-2, 3:-1]+8*X[3:-1, 2:-2]+8*X[2:-2, 1:-3]
        - 6*X[3:-1, 3:-1]-6*X[3:-1, 1:-3]
        - 2*X[2:-2, :-4]-2*X[2:-2, 4:]-2*X[4:, 2:-2]
        + 2*X[3:-1, 4:]+2*X[4:, 3:-1]+2*X[4:, 1:-3]+2*X[3:-1, :-4]
        - X[4:, 4:]-X[4:, :-4]-12*X[2:-2, 2:-2])
    Dl = (
        + 8*X[3:-1, 2:-2]+8*X[2:-2, 1:-3]+8*X[1:-3, 2:-2]
        - 6*X[3:-1, 1:-3]-6*X[1:-3, 1:-3]
        - 2*X[:-4, 2:-2]-2*X[4:, 2:-2]-2*X[2:-2, :-4]
        + 2*X[4:, 1:-3]+2*X[3:-1, :-4]+2*X[1:-3, :-4]+2*X[:-4, 1:-3]
        - X[4:, :-4]-X[:-4, :-4]-12*X[2:-2, 2:-2])
    # quantize
    Yu = np.clip(matlab.round(Du / q), -T, T)
    Yb = np.clip(matlab.round(Db / q), -T, T)
    Yl = np.clip(matlab.round(Dl / q), -T, T)
    Yr = np.clip(matlab.round(Dr / q), -T, T)

    # spam14
    g['spam14h'] = (
        + CoocN(np.vstack([Yu, Yb]), 'hor', T)
        + CoocN(np.hstack([Yl, Yr]), 'ver', T)
    )
    if directional:
        g['spam14v'] = (
            + CoocN(np.vstack([Yl, Yr]), 'hor', T)
            + CoocN(np.hstack([Yu, Yb]), 'ver', T)
        )

    # minmax24
    Dmin1 = np.minimum(Yu, Yl)
    Dmin2 = np.minimum(Yb, Yr)
    Dmin3 = np.minimum(Yu, Yr)
    Dmin4 = np.minimum(Yb, Yl)
    Dmax1 = np.maximum(Yu, Yl)
    Dmax2 = np.maximum(Yb, Yr)
    Dmax3 = np.maximum(Yu, Yr)
    Dmax4 = np.maximum(Yb, Yl)
    g['min24'] = CoocN(np.vstack([Dmin1, Dmin2, Dmin3, Dmin4]), 'hor', T)
    g['max24'] = CoocN(np.vstack([Dmax1, Dmax2, Dmax3, Dmax4]), 'hor', T)
    if directional:
        g['min24'] += CoocN(np.hstack([Dmin1, Dmin2, Dmin3, Dmin4]), 'ver', T)
        g['max24'] += CoocN(np.hstack([Dmax1, Dmax2, Dmax3, Dmax4]), 'ver', T)

    # minmax22
    UEq_min = np.minimum(Yu, Yb)
    REq_min = np.minimum(Yr, Yl)
    UEq_max = np.maximum(Yu, Yb)
    REq_max = np.maximum(Yr, Yl)
    g['min22h'] = (CoocN(UEq_min, 'hor', T) + CoocN(REq_min, 'ver', T))
    g['max22h'] = (CoocN(UEq_max, 'hor', T) + CoocN(REq_max, 'ver', T))
    if directional:
        g['min22v'] = (CoocN(REq_min, 'hor', T) + CoocN(UEq_min, 'ver', T))
        g['max22v'] = (CoocN(REq_max, 'hor', T) + CoocN(UEq_max, 'ver', T))

    # minmax41
    Dmin5 = np.minimum(Dmin1, Dmin2)
    Dmax5 = np.maximum(Dmax1, Dmax2)
    g['min41'] = CoocN(Dmin5, 'hor', T)
    g['max41'] = CoocN(Dmax5, 'hor', T)
    if directional:
        g['min41'] += CoocN(Dmin5, 'ver', T)
        g['max41'] += CoocN(Dmax5, 'ver', T)

    #
    return g
