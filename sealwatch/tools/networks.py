
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
import urllib.request


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_if_missing(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
    return dest


class Grayscale(transforms.Grayscale):
    def forward(self, img):
        if img.shape[0] == 1:
            return img
        elif img.shape[0] == 4:
            return img[3:]
        else:
            return super().forward(img)
