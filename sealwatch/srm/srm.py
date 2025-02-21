
from collections import OrderedDict
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, List, Dict

from .cooccurrence import all1st, all2nd, all3x3, all3rd, all5x5
from . import symm


def extract_from_file(
    path: Union[str, Path],
    **kw,
) -> Dict[str, np.ndarray]:
    # load
    img = np.array(Image.open(path))
    return extract(img, **kw)


def extract(
    x: np.ndarray,
    *,
    qs: List[List[int]] = [[1, 2], [1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2]],
    directional: bool = True,
) -> Dict[str, np.ndarray]:
    """Extracts spatial rich model for steganalysis.

    :param x: 2D input image
    :type x: np.ndarray
    :return: structured SRM features
    :rtype: OrderedDict
    """
    assert len(x.shape) == 2, 'single channel expected'
    assert len(qs) <= 5, f'invalid quantization steps {qs=}'
    while len(qs) < 5:
        qs.append([])
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
    #
    return result




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
