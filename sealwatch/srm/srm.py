
from collections import OrderedDict
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, List, Dict

from .cooccurrence import all1st, all2nd, all3x3, all3rd, all5x5
from . import symm
from .. import tools


# Order of the submodels of each family, as in the original Matlab implementation.
# The spam11 of 3x3 and 5x5 are merged into s35_spam11, placed with 5x5.
FAMILIES = ['s1', 's2', 's3', 's3x3', 's5x5']
SUBMODEL_ORDER = {
    's1': ['minmax22h', 'minmax34h', 'minmax22v', 'minmax24', 'minmax34v', 'minmax41', 'minmax34', 'minmax48h', 'minmax48v', 'minmax54', 'spam14hv'],
    's2': ['minmax21', 'minmax41', 'minmax32', 'minmax24h', 'minmax24v', 'spam12hv'],
    's3': ['minmax22h', 'minmax34h', 'minmax22v', 'minmax24', 'minmax34v', 'minmax41', 'minmax34', 'minmax48h', 'minmax48v', 'minmax54', 'spam14hv'],
    's3x3': ['minmax24', 'minmax22h', 'minmax22v', 'minmax41', 'spam11', 'spam14hv'],
    's5x5': ['minmax24', 'minmax22h', 'minmax22v', 'minmax41', 'spam11', 'spam14hv'],
}


def matlab_order(
    features: Dict[str, np.ndarray],
    qs: List[List[float]],
) -> Dict[str, np.ndarray]:
    """Orders the submodels as the original Matlab implementation.

    By family, then quantization step in the order of qs, then submodel.

    :param features: SRM features
    :param qs: quantization steps of each family
    :return: ordered SRM features
    """
    def rank(name: str):
        family, submodel, q = name.split('_')
        family = 's5x5' if family == 's35' else family
        f = FAMILIES.index(family)
        q_names = [f'q{q}'.replace('.', '') for q in qs[f]]
        order = SUBMODEL_ORDER[family]
        return (f, q_names.index(q), order.index(submodel) if submodel in order else len(order))

    return OrderedDict(sorted(features.items(), key=lambda item: rank(item[0])))


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> Dict[str, np.ndarray]:
    # load
    img = np.array(Image.open(path))
    return extract(img, **kw)


def _extract(
    x: np.ndarray,
    qs: List[List[float]],
) -> Dict[str, np.ndarray]:
    """Python implementation of SRM, see extract()."""
    x = x.astype(int)
    result = OrderedDict()

    # 1st order
    for q in qs[0]:
        result = symm.post_process(all1st(x, q*1), 'f1', q, result)
    # 2nd order
    for q in qs[1]:
        result = symm.post_process(all2nd(x, q*2), 'f2', q, result)
    # 3rd order
    for q in qs[2]:
        result = symm.post_process(all3rd(x, q*3), 'f3', q, result)
    # 3x3
    for q in qs[3]:
        result = symm.post_process(all3x3(x, q*4), 'f3x3', q, result)
    # 5x5
    for q in qs[4]:
        result = symm.post_process(all5x5(x, q*12), 'f5x5', q, result)
    # deterministic order, as in Matlab
    return matlab_order(result, qs)


def extract(
    x: np.ndarray,
    *,
    qs: List[List[float]] = [[1, 2], [1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2]],
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Extracts spatial rich model for steganalysis.

    :param x: grayscale image of shape [height, width], in [0, 255]
    :type x: np.ndarray
    :param qs: quantization steps of the five residual families,
        1st, 2nd, 3rd order, 3x3, 5x5
    :type qs: list of list of float
    :return: structured SRM features, ordered as the original Matlab implementation
    :rtype: OrderedDict

    :Examples:

    By default, this function uses the Python implementation. To use the (substantially faster) Rust implementation, type

    >>> with sw.BACKEND_RUST:
    >>>     features = sw.srm.extract(x)
    """
    assert len(x.shape) == 2, 'single channel expected'
    assert len(qs) <= 5, f'invalid quantization steps {qs=}'
    # copy, missing families without quantization steps
    qs = [list(q) for q in qs] + [[] for _ in range(5 - len(qs))]

    # choose implementation, Python if Rust is not available or does not support the input
    backend = tools.get_backend()
    rs = tools.rust_module('srm')
    rust_supported = x.dtype == np.uint8 and min(x.shape) >= 5
    if backend == tools.BACKEND_RUST and rs is not None and rust_supported:
        q_names = [[f'{q}'.replace('.', '') for q in qs_f] for qs_f in qs]
        return OrderedDict(rs.extract(x1=x, qs=qs, q_names=q_names))
    elif backend in {tools.BACKEND_RUST, tools.BACKEND_PYTHON}:
        return _extract(x, qs)
    else:
        raise NotImplementedError(f'unknown backend {backend}')




    # features = OrderedDict()

    # # 1st order
    # for q in qs[0]:

    #     features_all1st = all1st(x, q=q, directional=directional)
    #     features.update(post_process(features_all1st, prefix="s1", suffix=f"q{q}".replace(".", "")))
    # # 2nd order
    # for q in qs[1]:
    #     features_all2nd = all2nd(x, q=q*2, directional=directional)
    #     features.update(post_process(features_all2nd, prefix="s2", suffix=f"q{q}".replace(".", "")))

    # #     # features = post_process(all2nd(x, q*2), 'f2', q, features)
    # # 3rd order
    # for q in qs[2]:
    #     features_all3rd = all3rd(x, q=q*3, directional=directional)
    #     features.update(post_process(features_all3rd, prefix="s3", suffix=f"q{q}".replace(".", "")))
    #     # features = post_process(all3rd(x, q*3), 'f3', q, features)
    # # 3x3
    # for q in qs[3]:
    #     features_all3x3 = all3x3(x, q=q*4, directional=directional)
    #     features.update(post_process(features_all3x3, prefix="s3x3", suffix=f"q{q}".replace(".", "")))
    #     # features = post_process(all3x3(x, q*4), 'f3x3', q, features)
    # # 5x5
    # for q in qs[4]:
    #     features_all5x5 = all5x5(x, q=q*12, directional=directional)
    #     features.update(post_process(features_all5x5, prefix="s5x5", suffix=f"q{q}".replace(".", "")))
    #     # features = post_process(all5x5(x, q*12), 'f5x5', q, features)
    # # #
    # return features


# if __name__ == "__main__":
#     cover_filepath = "/home/bene/data/BOSSbase_1.01/jpegs_q75/1.jpeg"
#     extract_from_file(cover_filepath, directional=True)
