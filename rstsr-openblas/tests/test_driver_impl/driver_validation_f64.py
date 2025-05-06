# # Driver tests in Python

import numpy as np
import scipy


def fingerprint(a):
    return np.dot(np.cos(np.arange(a.size)), np.asarray(a, order="C").ravel())


# Path of npy files in rstsr-test-manifest

root = "../../../rstsr-test-manifest/resources/"

# ## make sure of random generation

a_raw = np.load(f"{root}/a-f64.npy")
b_raw = np.load(f"{root}/b-f64.npy")

assert np.isclose(fingerprint(a_raw), 191.28900005103065)
assert np.isclose(fingerprint(b_raw), -51.11100342180723)

# ## eigh driver tests

# ### dsyev*

a = a_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsyevd(a, lower=True)
assert np.isclose(fingerprint(w), -71.4747209499407)
assert np.isclose(fingerprint(np.abs(v)), -9.903934930318247)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsyevd(a, lower=False)
assert np.isclose(fingerprint(w), -71.4902453763506)
assert np.isclose(fingerprint(np.abs(v)), 6.973792268793419)
fingerprint(w), fingerprint(np.abs(v))

# ### dsygv*

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsygvd(a, b, uplo='L')
assert np.isclose(fingerprint(w), -89.60433120129908)
assert np.isclose(fingerprint(np.abs(v)), -5.243112559130817)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsygvd(a, b, uplo='U')
assert np.isclose(fingerprint(w), -65.27252612342873)
assert np.isclose(fingerprint(np.abs(v)), -7.0849504857534535)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsygvd(a, b, uplo='L', itype=2)
assert np.isclose(fingerprint(w), -2437.094304861363)
assert np.isclose(fingerprint(np.abs(v)), -4.108281604767547)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v, _ = scipy.linalg.lapack.dsygvd(a, b, uplo='L', itype=3)
assert np.isclose(fingerprint(w), -2437.094304861363)
assert np.isclose(fingerprint(np.abs(v)), 30.756098926747757)
fingerprint(w), fingerprint(np.abs(v))

# ## solve driver tests

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy()[:1024*512].reshape(1024, 512)
lu, piv, x, _ = scipy.linalg.lapack.dgesv(a, b)
assert np.isclose(fingerprint(lu), 5397.198541468395)
assert np.isclose(fingerprint(piv), -14.694714160751573)
assert np.isclose(fingerprint(x), -1951.253447757597)
fingerprint(lu), fingerprint(piv), fingerprint(x)




