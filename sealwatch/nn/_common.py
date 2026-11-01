
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Dict, Union
import urllib.request


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_if_missing(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        # download to temporary file, so that interrupted download is not cached
        part = dest.with_name(dest.name + '.part')
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, filename=part, reporthook=t.update_to)
        part.replace(dest)
    return dest


def keep_missing_buffers(module: torch.nn.Module, state_dict: Dict, prefix: str, *args) -> None:
    """Load pre-hook, keeps the current values of buffers missing in the state dict.

    Used for fixed filters. If a checkpoint contains them, they are loaded,
    otherwise the predefined ones of a newly constructed model are kept.
    Only the module's own buffers are considered, not those of submodules.

    Register with module.register_load_state_dict_pre_hook(keep_missing_buffers).
    """
    for name, buffer in module.named_buffers(recurse=False):
        state_dict.setdefault(prefix + name, buffer.detach().clone())


def load_state_dict(source: Union[str, Path]) -> Dict:
    """Loads weights from a local file or URL.

    URLs are downloaded once into the torch hub cache.
    Only tensors and containers are unpickled (weights_only).

    :param source: local path or URL
    :type source: str or pathlib.Path
    :return: loaded checkpoint
    :rtype: dict
    """
    source = str(source)
    if source.startswith(('http://', 'https://')):
        cache_dir = Path(torch.hub.get_dir()) / 'sealwatch'
        path = download_if_missing(source, cache_dir / source.split('/')[-1])
    else:
        path = Path(source)
    return torch.load(path, map_location='cpu', weights_only=True)


def infere_single(
    x: np.ndarray,
    model: torch.nn.Module,
    *,
    device: torch.device = None,
    normalize: bool = True,
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: grayscale image of shape [H, W], or color image of shape [H, W, C]
    :type x: np.ndarray
    :param model: model in eval mode, returning two logits (cover, stego)
    :type model: torch.nn.Module
    :param device: torch device to move the model to,
        by default the model stays where it is
    :type device: torch.device
    :param normalize: if True, divide the values by 255, otherwise the model takes [0, 255]
    :type normalize: bool
    :return: stego probability
    :rtype: np.ndarray
    """
    if model.training:
        raise ValueError('model must be in eval mode, call model.eval() first')
    # prepare data
    x = np.asarray(x) / 255. if normalize else np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None]
    elif x.ndim == 3:
        x = x.transpose(2, 0, 1)
    else:
        raise ValueError(f'expected image of shape [H, W] or [H, W, C], got {x.shape}')
    # match the model
    if device is not None:
        model = model.to(device)
    param = next(model.parameters())
    x_ = torch.from_numpy(x[None]).to(dtype=param.dtype, device=param.device)

    #
    with torch.no_grad():
        # infere
        logit = model(x_)
        y_ = torch.nn.functional.softmax(logit, dim=1)[0, 1]
        # convert back to numpy
        y = y_.detach().cpu().numpy()
    #
    return y


class Grayscale(transforms.Grayscale):
    def forward(self, img):
        if img.shape[0] == 1:
            return img
        elif img.shape[0] == 4:
            return img[3:]
        else:
            return super().forward(img)
