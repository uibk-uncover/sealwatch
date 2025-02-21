"""This module implements extraction of color rich models (CRM).

CRMs are handcrafted features, with * submodels, with total length *.
CRMs were introduced in

    [1] Fridrich, Kodovsky: Rich Models for Steganalysis of Digital Images. TIFS, 2011.

This code produces equivalent results compared to the paper authors' own Matlab implementation.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from collections import OrderedDict
import collections
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Union, List

from .cooccurrence import all1st, all2nd, all3rd, all3x3, all5x5
from .cooccurrence import Implementation
from .cooccurrence import cooccurrence3_col as Cooc3
from . import symm
from . import srm


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> Dict[str, np.ndarray]:
    img = np.array(Image.open(path))
    return extract(img, **kw)


def extract(
    x: np.ndarray,
    *,
    q: int = 1,
    Tc: int = 2,
    implementation: Implementation = Implementation.CRM_FIX_MIN24,
) -> Dict[str, np.ndarray]:
    """Extracts color rich model for steganalysis.

    :param x: 2D input image
    :type x: np.ndarray
    :return: structured CRM features
    :rtype: collections.OrderedDict
    """
    x = x.astype('float32')
    result = collections.OrderedDict()
    #
    construct_kw = {'T': Tc, 'CoocN': Cooc3, 'directional': False}
    process_kw = {
        'q': q,
        'Tc': Tc,
        'name_mapper': {
            'spam11': 'spam11c', 'spam12h': 'spam12c', 'spam14h': 'spam14c',
            'min21': 'min21c', 'max21': 'max21c',
            'min22h': 'min22c', 'max22h': 'max22c',
            'min24': 'min24c', 'max24': 'max24c', 'min24h': 'min24c', 'max24h': 'max24c',
            'min32': 'min32c', 'max32': 'max32c',
            'min34': 'min34c', 'max34': 'max34c', 'min34h': 'min34hvc', 'max34h': 'max34hvc',
            'min41': 'min41c', 'max41': 'max41c',
            'min48h': 'min48c', 'max48h': 'max48c',
            'min54': 'min54c', 'max54': 'max54c',
        }
    }

    # 1st order
    result = post_process(
        all1st(x, q*1, **construct_kw),
        f='f1',
        result=result,
        **process_kw,
    )
    # 2nd order
    result = post_process(
        all2nd(x, q*2, **construct_kw),
        f='f2',
        result=result,
        **process_kw,
    )
    # 3rd order
    result = post_process(
        all3rd(x, q*3, **construct_kw),
        f='f3',
        result=result,
        **process_kw,
    )
    # 3x3
    result = post_process(
        all3x3(x, q*4, **construct_kw),
        f='f3x3',
        result=result,
        **process_kw,
    )
    # 5x5
    result = post_process(
        all5x5(x, q*12, **construct_kw),
        f='f5x5',
        result=result,
        **process_kw,
    )
    #
    return result


def symm4_dir(  # order = 4
    f: np.ndarray,
    T: int,
) -> np.ndarray:
    """Marginalization by sign and directional symmetry for a feature vector
    stored as one of our 2*(2T+1)^order-dimensional feature vectors.
    This routine should be used for features possiessing sign and directional symmetry,
    such as spam-like features or 3x3 features. It should NOT be used
    for features from MINMAX residuals.
    Use the alternative symfea_minmax for this purpose.
    The feature f is assumed to be a 2dim x database_size matrix of features
    stored as columns (e.g., hor+ver, diag+minor_diag), with dim=2(2T+1)^order.
    """
    #
    B = 2*T+1
    c = B**4
    print(f.shape[0], c)
    assert f.shape[0] == 2*c
    # Dimension of the red(uced) / marginalized set
    red = B**4 - 2*T*(T+1)*B**2
    cube_min = f[:c].reshape((B, B, B, B), order='F')
    cube_max = f[c:].reshape((B, B, B, B), order='F')
    A = cube_min + np.flip(cube_max)
    #
    done = np.zeros(A.shape, dtype=bool)
    As = np.zeros(red, dtype=A.dtype)
    m = 0
    for i in range(-T, T+1):
        for j in range(-T, T+1):
            for k in range(-T, T+1):
                for n in range(-T, T+1):
                    # asymmetric bin
                    if i != n or j != k:
                        if not done[i+T, j+T, k+T, n+T]:
                            As[m] = A[i+T, j+T, k+T, n+T] + A[n+T, k+T, j+T, i+T]  # mirror-bins merged
                            done[i+T, j+T, k+T, n+T] = done[n+T, k+T, j+T, i+T] = True
                            m += 1
                    # symmetric bin
                    else:
                        As[m] = A[i+T, j+T, k+T, n+T]
                        done[i+T, j+T, k+T, n+T] = True
                        m += 1
    return As


def post_process(
    DATA: np.ndarray,
    f: str,
    q: int,
    result: collections.OrderedDict = None,
    Tc: int = 3,
    name_mapper: Dict[str, str] = {},
) -> Dict[str, np.ndarray]:
    """"""
    if result is None:
        result = collections.OrderedDict()
    for k in DATA:
        q_str = f'{q}'.replace('.', '')
        result[f'{f}_{name_mapper.get(k, k)}_q{q_str}'] = DATA[k].astype('float32')
    #
    for k in list(result.keys()):
        if k[0] == 's':
            continue
        T, N, Q = k.split('_')

        # minmax symmetrization
        if N[:3] in {'min', 'max'}:
            out = f's{T[1:]}_minmax{N[3:]}_{Q}'
            if out in result:
                continue
            Fmin = result[k.replace('max', 'min')].flatten(order='F')
            Fmax = result[k.replace('min', 'max')].flatten(order='F')
            if N[-1] == 'c':  # color symmetrization
                result[out] = symm.symm3_dir(np.hstack([Fmin, Fmax]), Tc).astype('float32')
            else:
                result[out] = symm4_dir(np.hstack([Fmin, Fmax]), 2)

        # spam symmetrization
        elif N[:4] == 'spam':
            out = f's{T[1:]}_{N}_{Q}'
            if out in result:
                continue
            if N[-1] == 'c':  # color symmetrization
                result[out] = symm.symm3(result[k], T=Tc).astype('float32')
            else:
                result[out] = symm.symm4(result[k], T=2)

    # delete result.f*
    result = collections.OrderedDict([
        (k, v) for k, v in result.items() if not k.startswith('f')
    ])

    # merge spam features
    KEYS = set(result.keys())
    for k in KEYS:
        T, N, Q = k.split('_')
        if N[:4] != 'spam':
            continue
        if T == '':
            continue
        if N[-1] == 'v' or (N == 'spam11' and T == 's5x5'):
            pass
        elif N[-1] == 'h':
            # h+v union
            out = f'{T}_{N}v_{Q}'
            if out in result:
                continue
            k2 = k.replace('h_', 'v_')
            Fh, Fv = result[k], result[k2]
            result[out] = np.hstack([Fh, Fv])
            del result[k]
            del result[k2]
        elif N in {'spam11', 'spam11c'}:
            # KBKV creation
            out = f's35_{N}_{Q}'
            if out in result:
                continue
            name1 = k.replace('5x5', '3x3')
            name2 = k.replace('3x3', '5x5')
            if name1 not in KEYS:
                continue
            if name2 not in KEYS:
                continue
            F_KB = result[name1]
            F_KV = result[name2]
            result[out] = np.hstack([F_KB, F_KV])
            del result[name1]
            del result[name2]

    return result
