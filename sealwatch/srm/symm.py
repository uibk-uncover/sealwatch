
from collections import OrderedDict
import numpy as np
import re
from typing import Dict


# def parse_feature_name(name):
#     match = re.search(f"^([a-z0-9]+)_([a-z0-9_]+)_(q[0-9]+)$", name)
#     prefix = match.group(1)
#     submodel_name = match.group(2)
#     postfix = match.group(3)

#     return prefix, submodel_name, postfix


# def post_process(
#     input_features: OrderedDict,
#     prefix: str,
#     suffix: str,
# ) -> Dict[str, np.ndarray]:
#     """
#     :param input_features: ordered dict
#     :param prefix
#     :param suffix
#     :return: ordered dict
#     """

#     # Copy input dict to avoid in-place modifications
#     input_features = OrderedDict(input_features)

#     # Set up another dictionary for the output features
#     output_features = OrderedDict()

#     #
#     # min/max features: Concatenate "min" and "max" to "minmax"
#     #
#     # Find min submodels
#     min_submodels = list(filter(lambda f: f.startswith("min"), input_features.keys()))

#     # Find the corresponding max models
#     max_submodels = list(map(lambda f: f.replace("min", "max"), min_submodels))

#     for min_submodel, max_submodel in zip(min_submodels, max_submodels):
#         assert min_submodel in input_features
#         assert max_submodel in input_features

#         features_min_submodel = input_features.pop(min_submodel).flatten(order="F").astype(np.float32)
#         features_max_submodel = input_features.pop(max_submodel).flatten(order="F").astype(np.float32)

#         output_submodel_name = f"{prefix}_" + min_submodel.replace("min", "minmax") + f"_{suffix}"
#         output_features[output_submodel_name] = symm4_dir(np.hstack([features_min_submodel, features_max_submodel]), T=2)

#     #
#     # SPAM features
#     #
#     spam_h_submodels = list(filter(lambda f: re.search("spam[0-9]+_hor", f), input_features.keys()))
#     for spam_h_submodel in spam_h_submodels:
#         # Symmetrize the spam*h submodel
#         features_spam_h_submodel = input_features.pop(spam_h_submodel).astype(np.float32)
#         features_spam_h_submodel = symm4(features_spam_h_submodel, T=2)

#         # Corresponding spam*v submodel
#         spam_v_submodel = re.sub("(spam[0-9]+_)hor", r"\1ver", spam_h_submodel)
#         # print(spam_v_submodel, input_features.keys())
#         assert spam_v_submodel in input_features
#         features_spam_v_submodel = input_features.pop(spam_v_submodel).astype(np.float32)
#         features_spam_v_submodel = symm4(features_spam_v_submodel, T=2)

#         # Concatenate the h and v parts, rename to "hv"
#         output_submodel_name = f"{prefix}_" + re.sub("(spam[0-9]+)h", r"\1hv", spam_h_submodel) + f"_{suffix}"
#         output_features[output_submodel_name] = np.hstack((features_spam_h_submodel, features_spam_v_submodel))

#     return output_features

def post_process(
    DATA: np.ndarray,
    f: str,
    q: int,
    result: OrderedDict = None,
    Tc: int = 3,
    name_mapper: Dict[str, str] = {},
) -> Dict[str, np.ndarray]:
    """"""
    if result is None:
        result = OrderedDict()
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
                result[out] = symm3_dir(np.hstack([Fmin, Fmax]), Tc).astype('float32')
            else:
                result[out] = symm4_dir(np.hstack([Fmin, Fmax]), 2)

        # spam symmetrization
        elif N[:4] == 'spam':
            out = f's{T[1:]}_{N}_{Q}'
            if out in result:
                continue
            if N[-1] == 'c':  # color symmetrization
                result[out] = symm3(result[k], T=Tc).astype('float32')
            else:
                result[out] = symm4(result[k], T=2)

    # delete result.f*
    result = OrderedDict([
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


def symm3(
    A: np.ndarray,
    T: int,
) -> np.ndarray:
    #
    B = 2*T + 1
    c = B**3
    assert A.size == c
    #
    red = 1 + 3*T + 4*T**2 + 2*T**3
    As = np.zeros(red, dtype=A.dtype)
    done = np.zeros_like(A, dtype=bool)

    # The only non-marginalized bin is the origin [0, 0, 0, 0]
    next_idx = 1
    As[0] = A[T, T, T]
    for i in range(-T, T + 1):
        for j in range(-T, T + 1):
            for k in range(-T, T + 1):
                if (not done[i+T, j+T, k+T]) and (abs(i) + abs(j) + abs(k) != 0):
                    As[next_idx] = A[T+i, T+j, T+k] + A[T-i, T-j, T-k]
                    done[T+i, T+j, T+k] = done[T-i, T-j, T-k] = True
                    if (i != k) and not done[T+k, T+j, T+i]:
                        As[next_idx] = As[next_idx] + A[T+k, T+j, T+i] + A[T-k, T-j, T-i]
                        done[T+k, T+j, T+i] = done[T-k, T-j, T-i] = True
                    next_idx += 1
    #
    return As


def symm4(
    A: np.ndarray,
    T: int,
) -> np.ndarray:
    """Symmetry marginalization routine. The purpose is to reduce the feature dimensionality and make the features more populated.
    It can be applied to 1D -- 5D co-occurrence matrices (order \in {1,2,3,4,5}) with sign and directional symmetries (explained below).

    Marginalization by symmetry pertains to the fact that, fundamentally, the differences between consecutive pixels in a natural image (both cover and stego) d1, d2, d3, ..., have the same probability of occurrence as the triple -d1, -d2, -d3, ...
    Directional marginalization pertains to the fact that the differences d1, d2, d3, ... in a natural (cover and stego) image are as likely to occur as ..., d3, d2, d1.
    :param A: ndarray of shape (2T + 1)^order
    :param T: threshold
    :param order: co-occurrence order
    :return: symmetrized 1-D array of shape [num_ouput_features]
    """
    # Skip index 0, where the origin is stored
    next_idx = 1
    B = 2 * T + 1
    done = np.zeros_like(A, dtype=bool)

    # assert order == 4
    assert A.size == B ** 4
    As = np.zeros(B**2 + 4*T**2*(T+1)**2, dtype=A.dtype)

    # The only non-marginalized bin is the origin [0, 0, 0, 0]
    As[0] = A[T, T, T, T]
    for i in range(-T, T + 1):
        for j in range(-T, T + 1):
            for k in range(-T, T + 1):
                for n in range(-T, T + 1):
                    if (not done[i+T, j+T, k+T, n+T]) and (abs(i) + abs(j) + abs(k) + abs(n) != 0):
                        As[next_idx] = A[T+i, T+j, T+k, T+n] + A[T-i, T-j, T-k, T-n]
                        done[T+i, T+j, T+k, T+n] = done[T-i, T-j, T-k, T-n] = True
                        if ((i != n) or (j != k)) and not done[T+n, T+k, T+j, T+i]:
                            As[next_idx] = As[next_idx] + A[T+n, T+k, T+j, T+i] + A[T-n, T-k, T-j, T-i]
                            done[T+n, T+k, T+j, T+i] = done[T-n, T-k, T-j, T-i] = True
                        next_idx += 1
    #
    return As


def symm3_dir(  # order = 3
    f: np.ndarray,
    T: int,
) -> np.ndarray:
    """"""
    #
    B = 2*T+1
    c = B**3
    assert f.shape[0] == 2*c
    # Dimension of the red(uced) / marginalized set
    red = B**3 - T*B**2
    cube_min = f[:c].reshape((B, B, B), order='F')
    cube_max = f[c:].reshape((B, B, B), order='F')
    A = cube_min + np.flip(cube_max)
    #
    done = np.zeros(A.shape, dtype=bool)
    As = np.zeros(red, dtype=A.dtype)
    m = 0
    for i in range(-T, T+1):
        for j in range(-T, T+1):
            for k in range(-T, T+1):
                # asymmetric bin
                if i != k:
                    if not done[i+T, j+T, k+T]:
                        As[m] = A[i+T, j+T, k+T] + A[k+T, j+T, i+T]  # mirror-bins merged
                        done[i+T, j+T, k+T] = done[k+T, j+T, i+T] = True
                        m += 1
                # symmetric bin
                else:
                    As[m] = A[i+T, j+T, k+T]
                    done[i+T, j+T, k+T] = True
                    m += 1
    return As


def symm4_dir(  # order = 4
    f: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Symmetry marginalization routine. The purpose is to reduce the feature dimensionality and make the features more populated.
    A is are two concatenated arrays of features, each of size (2*T+1)^order.

    Directional marginalization pertains to the fact that the differences d1, d2, d3, ... in a natural (both cover and stego) image are as likely to occur as ..., d3, d2, d1.

    Basically, we merge all pairs of bins (i,j,k, ...) and (..., k,j,i) as long as they are two different bins. Thus, instead of dim = (2T+1)^order, we decrease the dim by 1/2*{# of non-symmetric bins}.
    For order = 3, the reduced dim is (2T+1)^order - T(2T+1)^(order-1), for order = 4, it is (2T+1)^4 - 2T(T+1)(2T+1)^2.

    :param f: features
    :param T: truncation threshold
    :return: symmetrized features
    """

    # number of bins
    B = 2 * T + 1
    # input dimensionality: [B, B, B, B], just flattened
    c = B ** 4

    assert f.shape[0] == 2*c

    # Dimension of the red(uced) / marginalized set
    red = B ** 4 - 2 * T * (T + 1) * B ** 2

    # Split again into min and max features
    cube_min = f[:c].reshape((B, B, B, B), order='F')
    cube_max = f[c:].reshape((B, B, B, B), order='F')

    # Sign-symmetrize
    A = cube_min + np.flip(cube_max)

    # Use left-right symmetry
    done = np.zeros(A.shape, dtype=bool)
    As = np.zeros(red, dtype=A.dtype)
    m = 0
    for i in range(-T, T+1):
        for j in range(-T, T+1):
            for k in range(-T, T+1):
                for n in range(-T, T+1):
                    if i != n or j != k:
                        # asymmetric bin, merge two bins
                        if not done[i+T, j+T, k+T, n+T]:
                            As[m] = A[i+T, j+T, k+T, n+T] + A[n+T, k+T, j+T, i+T]  # mirror-bins merged
                            done[i+T, j+T, k+T, n+T] = done[n+T, k+T, j+T, i+T] = True
                            m += 1
                    else:
                        # symmetric bin, nothing to merge
                        As[m] = A[i+T, j+T, k+T, n+T]
                        done[i+T, j+T, k+T, n+T] = True
                        m += 1
    return As
