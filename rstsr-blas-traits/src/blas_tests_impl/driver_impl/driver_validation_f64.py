# # Driver tests in Python

import numpy as np
import os
import scipy


def fingerprint(a):
    return np.dot(np.cos(np.arange(a.size)), np.asarray(a, order="C").ravel())


# Path of npy files in rstsr-test-manifest, resolved relative to this script

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(script_dir, "../../../../rstsr-test-manifest/resources/")

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

# ### gesv

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy()[:1024*512].reshape(1024, 512)
lu, piv, x, _ = scipy.linalg.lapack.dgesv(a, b)
assert np.isclose(fingerprint(lu), 5397.198541468395)
assert np.isclose(fingerprint(piv), -14.694714160751573)
assert np.isclose(fingerprint(x), -1951.253447757597)
fingerprint(lu), fingerprint(piv), fingerprint(x)

# ### getrf, getri

a = a_raw.copy().reshape(1024, 1024)
lu, piv, _ = scipy.linalg.lapack.dgetrf(a)
assert np.isclose(fingerprint(lu), 5397.198541468395)
assert np.isclose(fingerprint(piv), -14.694714160751573)
fingerprint(lu), fingerprint(piv)

inv_a, _ = scipy.linalg.lapack.dgetri(lu, piv)
assert np.isclose(fingerprint(inv_a), 143.3900557703788)
fingerprint(inv_a)

# ### potrf

# Please note that driver implementation in rust does not clean upper/lower triangular.

b = b_raw.copy().reshape(1024, 1024)
c, _ = scipy.linalg.lapack.dpotrf(b, lower=True, clean=0)
assert np.isclose(fingerprint(c), 35.17266259472725)
fingerprint(c)

b = b_raw.copy().reshape(1024, 1024)
c, _ = scipy.linalg.lapack.dpotrf(b, lower=False, clean=0)
assert np.isclose(fingerprint(c), -53.53353704132017)
fingerprint(c)

# ### sysv

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy()[:1024*512].reshape(1024, 512)
udut, piv, x, _ = scipy.linalg.lapack.dsysv(a, b, lower=True)
assert np.isclose(fingerprint(udut), -1201.6472395568974)
assert np.isclose(fingerprint(piv), -16668.7094872639)
assert np.isclose(fingerprint(x), -397.12032355166446)
fingerprint(udut), fingerprint(piv), fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy()[:1024*512].reshape(1024, 512)
udut, piv, x, _ = scipy.linalg.lapack.dsysv(a, b, lower=False)
assert np.isclose(fingerprint(udut), 1182.7836118324408)
assert np.isclose(fingerprint(piv), 11905.503011559245)
assert np.isclose(fingerprint(x), -314.4502289190444)
fingerprint(udut), fingerprint(piv), fingerprint(x)

# ## svd driver tests

a = a_raw.copy()[:1024*512].reshape(1024, 512)
u, s, vt, _ = scipy.linalg.lapack.dgesvd(a)
assert np.isclose(fingerprint(np.abs(u)), -1.9368850983570982)
assert np.isclose(fingerprint(s), 33.969339071043095)
assert np.isclose(fingerprint(np.abs(vt)), 13.465522484136157)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

a = a_raw.copy()[:1024*512].reshape(1024, 512)
u, s, vt, _ = scipy.linalg.lapack.dgesvd(a, full_matrices=False)
assert np.isclose(fingerprint(np.abs(u)), -9.144981428076894)
assert np.isclose(fingerprint(s), 33.969339071043095)
assert np.isclose(fingerprint(np.abs(vt)), 13.465522484136157)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

# ## eig driver tests

# the packed real eigenvector matrix of a real driver is expanded to complex
# columns: v(j) = v[:, j] (+|-) 1j * v[:, j + 1] for a conjugate pair


def _unpack_real_eigvecs(v, wi):
    n = v.shape[0]
    out = np.zeros((n, n), dtype=complex)
    j = 0
    while j < n:
        if wi[j] == 0:
            out[:, j] = v[:, j]
            j += 1
        else:
            out[:, j] = v[:, j] + 1j * v[:, j + 1]
            out[:, j + 1] = v[:, j] - 1j * v[:, j + 1]
            j += 2
    return out


# ### dgeev

a = a_raw.copy().reshape(1024, 1024)
wr, wi, vl, vr, info = scipy.linalg.lapack.dgeev(a, compute_vl=1, compute_vr=1)
w = wr + 1j * wi
order = np.argsort(w)
vl = _unpack_real_eigvecs(vl, wi)
vr = _unpack_real_eigvecs(vr, wi)
assert np.isclose(fingerprint(np.sort(w)), -36.380652131982586-116.87071686473277j)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -15.906522814903258)
assert np.isclose(fingerprint(np.abs(vr[:, order])), -12.000277853410061)
fingerprint(np.sort(w)), fingerprint(np.abs(vl[:, order])), fingerprint(np.abs(vr[:, order]))

# ### zgeev

ac = np.load(f"{root}/a-c64.npy").reshape(1024, 1024)
w, vl, vr, info = scipy.linalg.lapack.zgeev(ac, compute_vl=1, compute_vr=1)
order = np.argsort(w)
assert np.isclose(fingerprint(np.sort(w)), -53.51484095091083-871.7616347408099j)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -5.456470679201179)
assert np.isclose(fingerprint(np.abs(vr[:, order])), 4.932401633755345)
fingerprint(np.sort(w)), fingerprint(np.abs(vl[:, order])), fingerprint(np.abs(vr[:, order]))

# ### dggev (256x256, B = M^T M + 256 I well-conditioned; LAPACK scaling kept)

# the real parts of a conjugate pair may differ in the last bit, so the order
# of the two members after sorting is not defined; fingerprint the sorted real
# parts and the sorted |imaginary parts| (robust), not the complex spectrum
m = a_raw.copy()[:256*256].reshape(256, 256)
b = m.T @ m + 256 * np.eye(256)
alphar, alphai, beta, vl, vr, work, info = scipy.linalg.lapack.dggev(m, b, compute_vl=1, compute_vr=1)
alpha = alphar + 1j * alphai
w = alpha / beta
order = np.argsort(w)
vl = _unpack_real_eigvecs(vl, alphai)
vr = _unpack_real_eigvecs(vr, alphai)
# the eigenvector order is backend-dependent and the raw scaling of the GGEV
# vectors differs between LAPACK builds (observed MKL vs OpenBLAS): rescale
# every column to unit 2-norm, then fingerprint the columns of |v| one by one
# and sort (order-invariant)


def _fp_vecs(v):
    v = v / np.linalg.norm(v, axis=0)
    fp = np.array([fingerprint(np.abs(v[:, j])) for j in range(v.shape[1])])
    return fingerprint(np.sort(fp))


assert np.isclose(fingerprint(np.sort(w.real)), -0.03604323300216897)
assert np.isclose(fingerprint(np.sort(np.abs(w.imag))), -0.026149942284387048)
assert np.isclose(_fp_vecs(vl), -0.8719042678843263)
assert np.isclose(_fp_vecs(vr), -1.225589423351893)
fingerprint(np.sort(w.real)), fingerprint(np.sort(np.abs(w.imag))), _fp_vecs(vl), _fp_vecs(vr)

# ### zggev (256x256 leading blocks, B = data + 256 I; LAPACK scaling kept)

ac = np.load(f"{root}/a-c64.npy")[:256*256].reshape(256, 256)
bc = np.load(f"{root}/b-c64.npy")[:256*256].reshape(256, 256) + 256 * np.eye(256)
alpha, beta, vl, vr, rwork, info = scipy.linalg.lapack.zggev(ac, bc, compute_vl=1, compute_vr=1)
w = alpha / beta
# the spectrum is fingerprinted as the independently sorted real and imaginary
# parts, the eigenvectors as above (order-invariant)
assert np.isclose(fingerprint(np.sort(w.real)), -0.11836546248737323)
assert np.isclose(fingerprint(np.sort(w.imag)), -0.11210131721328423)
assert np.isclose(_fp_vecs(vl), -1.3881039015638714)
assert np.isclose(_fp_vecs(vr), -1.9454332085201405)
fingerprint(np.sort(w.real)), fingerprint(np.sort(w.imag)), _fp_vecs(vl), _fp_vecs(vr)

# ## lstsq driver tests

# DGELSD with the rcond default of the entry layer (eps * max(m, n)); the
# workspace sizes come from the same LWORK query the thunk performs. The
# driver reports no residuals when rank < min(m, n), so the rank-deficient
# case is fingerprinted as NaN even though LAPACK leaves values in B(n+1:m).


def _gelsd(a, b, rcond):
    m, n = a.shape
    nrhs = b.shape[1]
    b_buf = np.zeros((max(m, n), nrhs), order="F")
    b_buf[:m] = b
    work, iwork, _ = scipy.linalg.lapack.dgelsd_lwork(m, n, nrhs, cond=rcond)
    x, s, rank, info = scipy.linalg.lapack.dgelsd(
        a.copy(order="F"), b_buf, lwork=int(work), size_iwork=int(iwork), cond=rcond
    )
    assert info == 0
    if m > n and rank == n:
        residuals = np.sum(x[n:m] ** 2, axis=0)
    elif m > n:
        residuals = np.full(nrhs, np.nan)
    else:
        residuals = np.zeros(0)
    return x[:n], s, rank, residuals


a = a_raw.copy()[:1024 * 512].reshape(1024, 512)
b = b_raw.copy()[:1024].reshape(1024, 1)
x, s, rank, residuals = _gelsd(a, b, np.finfo(np.float64).eps * 1024)
assert np.isclose(fingerprint(x), 0.6714195293796388)
assert np.isclose(fingerprint(s), 33.96933907104208)
assert np.isclose(fingerprint(residuals), 495.06381572902126)
assert rank == 512
fingerprint(x), fingerprint(s), rank, fingerprint(residuals)

a = a_raw.copy()[:1024 * 512].reshape(1024, 512)
b = b_raw.copy()[:1024 * 3].reshape(1024, 3)
x, s, rank, residuals = _gelsd(a, b, np.finfo(np.float64).eps * 1024)
assert np.isclose(fingerprint(x), -2.417996890572109)
assert np.isclose(fingerprint(s), 33.96933907104208)
assert np.isclose(fingerprint(residuals), 544.9834618414122)
assert rank == 512
fingerprint(x), fingerprint(s), rank, fingerprint(residuals)

a = a_raw.copy()[:512 * 1024].reshape(512, 1024)
b = b_raw.copy()[:512].reshape(512, 1)
x, s, rank, residuals = _gelsd(a, b, np.finfo(np.float64).eps * 1024)
assert np.isclose(fingerprint(x), 0.1316952775202293)
assert np.isclose(fingerprint(s), 32.27742168207754)
assert residuals.size == 0
assert rank == 512
fingerprint(x), fingerprint(s), rank

a = a_raw.copy()[:1024 * 512].reshape(1024, 512)
a[:, 2] = a[:, 0] + a[:, 1]
b = b_raw.copy()[:1024].reshape(1024, 1)
x, s, rank, residuals = _gelsd(a, b, np.finfo(np.float64).eps * 1024)
assert np.isclose(fingerprint(x), 0.6203201140848792)
assert np.isclose(fingerprint(s), 46.463897960241695)
assert np.isnan(residuals).all()
assert rank == 511
fingerprint(x), fingerprint(s), rank, fingerprint(residuals)
