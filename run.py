
import conseal as cl
from glob import glob
import imageops
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sealwatch as sw
from tqdm import tqdm


def hcfm(hcf, order: int = 1) -> float:
    return sum([
        (k/256)**order * hcf[k]
        for k in range(128)
    ])  # / np.sum(hcf)


def _hcfcom(
    x: np.ndarray,
    order: int = 1,
) -> float:
    # histogram characteristic function (HCF)
    h, bins = np.histogram(x, range(257), range=(0, 256), density=True)
    hcf = np.fft.fft(h)
    # center of mass (COM)
    hcf = np.abs(hcf[1:129])  # remove DC and right half
    return hcfm(hcf) / np.sum(hcf)
    # return sum([
    #     (k/256)**order * hcf[k]
    #     for k in range(128)
    # ]) / np.sum(hcf)


# def _hcfcom3d(
#     x: np.ndarray,
#     order: int = 1,
# ) -> float:
#     # histogram characteristic function (HCF)
#     bins = list(range(257))
#     h, edges = np.histogramdd(
#         x.reshape(-1, 3),
#         bins=(bins, bins, bins),
#         range=((0, 256), (0, 256), (0, 256)),
#         # density=True,
#     )
#     hcf = np.fft.fft(h)
#     # center of mass (COM)
#     hcf = np.abs(hcf[1:129, 1:129, 1:129])  # remove DC and right half
#     return np.array([
#         hcfm(np.sum(hcf, axis=tuple(set(range(3)) - {i}))) / np.sum(hcf)
#         for i in range(3)
#     ])

# def _hcfcom3d(
#     x: np.ndarray,
#     order: int = 1,
# ) -> float:
#     # histogram characteristic function (HCF)
#     bins = list(range(257))
#     h, edges = np.histogramdd(
#         x.reshape(-1, 3),
#         bins=(bins, bins, bins),
#         range=((0, 256), (0, 256), (0, 256)),
#         # density=True,
#     )
#     hcf = np.fft.fft(h)
#     # center of mass (COM)
#     hcf = np.abs(hcf[1:129, 1:129, 1:129])  # remove DC and right half
#     return np.array([
#         hcfm(np.sum(hcf, axis=tuple(set(range(3)) - {i}))) / np.sum(hcf)
#         for i in range(3)
#     ])


# x0 = np.array(Image.open(f'{os.environ["DATA"]}/data/alaska2/fabrika-2024-09-23/images_ahd/00002.png'))
# print(x0.shape)
# hcf = _hcfcom3d(x0)
# print(hcf)


alpha = .4
df = []
f0, f1 = [], []
for fname in tqdm(glob(f'{os.environ["DATA"]}/data/alaska2/fabrika-2024-09-23/images_ahd/*.png')[:500]):

    # load cover
    x0 = np.array(Image.open(fname).convert('L')).astype('float32')
    # x0_hcfcom = sw.features.hcfcom.extract_hcfcom(x0, order=1.5)  # extract HCF-COM
    x0_spam = sw.features.spam.extract_spam686_features_from_img(x0)
    # x0_hcfcom = _hcfcom3d(x0)  # extract HCF-COM
    # #
    # x0c = imageops.scale_image(x0, np.array(x0.shape[:2]) // 2, 'nearest', use_antialiasing=True)
    # x0c_hcfcom = _hcfcom3d(x0c)  # extract HCF-COM

    # embed
    x1 = cl.lsb.simulate(x0, alpha, modify=cl.LSB_MATCHING, seed=12345)
    # x1_hcfcom = sw.features.hcfcom.extract_hcfcom(x1, order=1.5)  # extract HCF-COM
    x1_spam = sw.features.spam.extract_spam686_features_from_img(x1)
    # x1_hcfcom = _hcfcom3d(x1)  # extract HCF-COM
    # #
    # x1c = imageops.scale_image(x1, np.array(x1.shape[:2]) // 2, 'nearest', use_antialiasing=True)
    # x1c_hcfcom = _hcfcom3d(x1c)  # extract HCF-COM

    #
    # f0.append(x0_hcfcom)
    # f1.append(x1_hcfcom)
    f0.append(sw.utils.grouping.flatten_single(x0_spam))
    f1.append(sw.utils.grouping.flatten_single(x1_spam))
    df.append({
        'fname': fname,
        'alpha': alpha,
    })

df = pd.DataFrame(df)
f0, f1 = np.array(f0), np.array(f1)
print(df)
print(f0.shape, f1.shape)

#
X = np.concat([f0, f1], axis=0)
y = np.concat([np.zeros(f0.shape[0]), np.ones(f1.shape[0])], axis=0)
print(X.shape, y.shape)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.5, random_state=42)
print(X_tr.shape, y_tr.shape)
print(X_te.shape, y_te.shape)
model = GaussianNB().fit(X_tr, y_tr)
y_tr_pred = model.predict(X_tr)
y_te_pred = model.predict(X_te)
print('tr accuracy', accuracy_score(y_tr, y_tr_pred))
print('te accuracy', accuracy_score(y_te, y_te_pred))


# fig, ax = plt.subplots()
# for i, df_group in df.groupby('fname'):
#     df_group = df_group.sort_values('alpha')
#     ax.scatter(df_group['alpha'], df_group['x1'])
# plt.show()