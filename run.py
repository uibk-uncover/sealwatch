
import numpy as np
from PIL import Image
import sealwatch as sw
import time

x0 = np.array(Image.open('test/assets/cover/uncompressed_gray/seal1.png'))

start = time.perf_counter()
with sw.tools.BACKEND_RUST:
    f_rs = sw.tools.flatten(sw.srm.extract(x=x0))
end = time.perf_counter()
print(f'srm [BACKEND_RUST]: {end-start:.04f} s')

start = time.perf_counter()
with sw.tools.BACKEND_PYTHON:
    f_py = sw.tools.flatten(sw.srm.extract(x=x0))
end = time.perf_counter()
print(f'srm [BACKEND_PYTHON]: {end-start:.04f} s')

np.testing.assert_array_equal(f_rs, f_py)
print('srm matches!')

start = time.perf_counter()
with sw.tools.BACKEND_RUST:
    f_rs = sw.tools.flatten(sw.srmq1.extract(x=x0))
end = time.perf_counter()
print(f'srmq1 [BACKEND_RUST]: {end-start:.04f} s')

start = time.perf_counter()
with sw.tools.BACKEND_PYTHON:
    f_py = sw.tools.flatten(sw.srmq1.extract(x=x0))
end = time.perf_counter()
print(f'srmq1 [BACKEND_PYTHON]: {end-start:.04f} s')

np.testing.assert_array_equal(f_rs, f_py)
print('srmq1 matches!')

start = time.perf_counter()
with sw.tools.BACKEND_RUST:
    f_rs = sw.tools.flatten(sw.spam.extract(x=x0))
end = time.perf_counter()
print(f'spam [BACKEND_RUST]: {end-start:.04f} s')

start = time.perf_counter()
with sw.tools.BACKEND_PYTHON:
    f_py = sw.tools.flatten(sw.spam.extract(x=x0))
end = time.perf_counter()
print(f'spam [BACKEND_PYTHON]: {end-start:.04f} s')

np.testing.assert_array_equal(f_rs, f_py)
print('spam matches!')

x0 = np.array(Image.open('test/assets/cover/uncompressed_color/seal1.png'))

start = time.perf_counter()
with sw.tools.BACKEND_RUST:
    f_rs = sw.tools.flatten(sw.crm.extract(x=x0))
end = time.perf_counter()
print(f'crm [BACKEND_RUST]: {end-start:.04f} s')

start = time.perf_counter()
with sw.tools.BACKEND_PYTHON:
    f_py = sw.tools.flatten(sw.crm.extract(x=x0))
end = time.perf_counter()
print(f'crm [BACKEND_PYTHON]: {end-start:.04f} s')

np.testing.assert_array_equal(f_rs, f_py)
print('crm matches!')


