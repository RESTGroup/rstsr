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

# ## tests

# ### cholesky

b = b_raw.copy().reshape(1024, 1024)
c = np.linalg.cholesky(b)
assert np.isclose(fingerprint(c), 43.21904478556176)
fingerprint(c)

b = b_raw.copy().reshape(1024, 1024)
c = scipy.linalg.cholesky(b, lower=False)
assert np.isclose(fingerprint(c), -25.925655124816647)
fingerprint(c)

# ### eigh

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eigh(a)
assert np.isclose(fingerprint(w), -71.4747209499407)
assert np.isclose(fingerprint(np.abs(v)), -9.903934930318247)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eigh(a, UPLO="U")
assert np.isclose(fingerprint(w), -71.4902453763506)
assert np.isclose(fingerprint(np.abs(v)), 6.973792268793419)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, lower=True)
assert np.isclose(fingerprint(w), -89.60433120129908)
assert np.isclose(fingerprint(np.abs(v)), -5.243112559130817)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, lower=False, type=3)
assert np.isclose(fingerprint(w), -2503.84161931662)
assert np.isclose(fingerprint(np.abs(v)), 152.17700520642055)
fingerprint(w), fingerprint(np.abs(v))

# ### eig

a = a_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eig(a)
# a non-Hermitian spectrum is unordered; sort to a canonical order so the
# reference is backend-independent (np.sort / np.argsort on complex sort
# lexicographically by real then imaginary)
order = np.argsort(w)
assert np.isclose(fingerprint(np.sort(w)), -36.380652131982586-116.87071686473277j)
fingerprint(np.sort(w))
# eigenvector columns reordered to the same canonical order, then the
# elementwise absolute value (phase- and sign-invariant) before fingerprinting
assert np.isclose(fingerprint(np.abs(v[:, order])), -12.000277853410061)
fingerprint(np.abs(v[:, order]))

# ### eigvals

a = a_raw.copy().reshape(1024, 1024)
w = np.linalg.eigvals(a)
assert np.isclose(fingerprint(np.sort(w)), -36.380652131982586-116.87071686473277j)
fingerprint(np.sort(w))

# ### eig (left / both)

a = a_raw.copy().reshape(1024, 1024)
w, vl = scipy.linalg.eig(a, left=True, right=False)
order = np.argsort(w)
assert np.isclose(fingerprint(np.sort(w)), -36.380652131982586-116.87071686473277j)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -15.906522814903258)
fingerprint(np.sort(w)), fingerprint(np.abs(vl[:, order]))

a = a_raw.copy().reshape(1024, 1024)
w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
order = np.argsort(w)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -15.906522814903258)
assert np.isclose(fingerprint(np.abs(vr[:, order])), -12.000277853410061)
fingerprint(np.abs(vl[:, order])), fingerprint(np.abs(vr[:, order]))


# ### eig (generalized A v = lambda B v; SciPy rescales the vectors to unit 2-norm)
# 256 x 256 leading blocks of the manifest arrays: a full-size QZ solve is
# far costlier than GEEV (the standard problem above keeps 1024 x 1024)

# the sorted complex order is not robust when distinct eigenvalues share a real
# part, so the spectrum is fingerprinted as the independently sorted real and
# imaginary parts, and the eigenvector block as the sorted per-column
# fingerprints of |v| (both order-invariant)
def _fp_spectrum(w):
    return fingerprint(np.sort(w.real)), fingerprint(np.sort(w.imag))


def _fp_vecs(v):
    fp = np.array([fingerprint(np.abs(v[:, j])) for j in range(v.shape[1])])
    return fingerprint(np.sort(fp))


a = a_raw[:256*256].copy().reshape(256, 256)
b = b_raw[:256*256].reshape(256, 256) @ b_raw[:256*256].reshape(256, 256).T + 256 * np.eye(256)  # SPD, a well-conditioned pencil

w, vr = scipy.linalg.eig(a, b)
assert np.isclose(_fp_spectrum(w)[0], -0.08043969532621999)
assert np.isclose(_fp_spectrum(w)[1], -0.07080695445332981)
assert np.isclose(_fp_vecs(vr), -1.5688154447109144)
_fp_spectrum(w), _fp_vecs(vr)

a = a_raw[:256*256].copy().reshape(256, 256)
b = b_raw[:256*256].reshape(256, 256) @ b_raw[:256*256].reshape(256, 256).T + 256 * np.eye(256)  # SPD, a well-conditioned pencil

w, vl, vr = scipy.linalg.eig(a, b, left=True, right=True)
assert np.isclose(_fp_vecs(vl), -1.2719566873337649)
assert np.isclose(_fp_vecs(vr), -1.5688154447109144)
_fp_vecs(vl), _fp_vecs(vr)

# ### eigvals (generalized)

a = a_raw[:256*256].copy().reshape(256, 256)
b = b_raw[:256*256].reshape(256, 256) @ b_raw[:256*256].reshape(256, 256).T + 256 * np.eye(256)  # SPD, a well-conditioned pencil

w = scipy.linalg.eigvals(a, b)
assert np.isclose(_fp_spectrum(w)[0], -0.08043969532621999)
assert np.isclose(_fp_spectrum(w)[1], -0.07080695445332981)
_fp_spectrum(w)

# ### inv

a = a_raw.copy().reshape(1024, 1024)
a_inv = np.linalg.inv(a)
assert np.isclose(fingerprint(a_inv), 143.39005577037764)
fingerprint(a_inv)

# ### solve_general

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = np.linalg.solve(a, b)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024].copy().reshape(1024)
x = np.linalg.solve(a, b)
fingerprint(x)

# ### sovle_symmetric

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="sym", lower=True)
assert np.isclose(fingerprint(x), -397.1203235513806)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="sym", lower=False)
assert np.isclose(fingerprint(x), -314.45022891879034)
fingerprint(x)

# ### sovle_triangular

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw.copy().reshape(1024, 1024)
x = scipy.linalg.solve(b, a, assume_a="lower triangular")
assert np.isclose(fingerprint(x), -2.6133848012216587)
fingerprint(x)

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw.copy().reshape(1024, 1024)
x = scipy.linalg.solve(b, a, assume_a="upper triangular")
assert np.isclose(fingerprint(x), 5.112256818100785)
fingerprint(x)

# ### slogdot

a = a_raw.copy().reshape(1024, 1024)
sgn, logabsdet = np.linalg.slogdet(a)
assert np.isclose(sgn, -1)
assert np.isclose(logabsdet, 3031.1259211802403)
sgn, logabsdet

# ### det

a = a_raw[:25].copy().reshape(5, 5)
det = np.linalg.det(a)
assert np.isclose(det, 3.9699917597338046)
det

np.linalg.slogdet(a)

# ### svd

a = a_raw[:1024*512].copy().reshape(1024, 512)
(u, s, vt) = scipy.linalg.svd(a)
assert np.isclose(fingerprint(np.abs(u)), -1.9368850983570982)
assert np.isclose(fingerprint(s), 33.969339071043095)
assert np.isclose(fingerprint(np.abs(vt)), 13.465522484136157)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

a = a_raw[:1024*512].copy().reshape(1024, 512)
(u, s, vt) = scipy.linalg.svd(a, full_matrices=False)
assert np.isclose(fingerprint(np.abs(u)), -9.144981428076894)
assert np.isclose(fingerprint(s), 33.969339071043095)
assert np.isclose(fingerprint(np.abs(vt)), 13.465522484136157)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

a = a_raw[:1024*512].copy().reshape(1024, 512)
s = scipy.linalg.svd(a, compute_uv=False)
assert np.isclose(fingerprint(s), 33.969339071043095)
fingerprint(s)

a = a_raw[:1024*512].copy().reshape(512, 1024)
(u, s, vt) = scipy.linalg.svd(a, full_matrices=False)
assert np.isclose(fingerprint(np.abs(u)), -3.716931052161584)
assert np.isclose(fingerprint(s), 32.27742168207757)
assert np.isclose(fingerprint(np.abs(vt)), -0.32301437281530243)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

# ### pinv

a = a_raw[:1024*512].copy().reshape(1024, 512)
a_pinv, rank = scipy.linalg.pinv(a, return_rank=True, atol=20, rtol=0.3)
assert np.isclose(fingerprint(a_pinv), 0.0878262837784408)
assert rank == 163
fingerprint(a_pinv), rank

a = a_raw[:1024*512].copy().reshape(512, 1024)
a_pinv, rank = scipy.linalg.pinv(a, return_rank=True, atol=20, rtol=0.3)
assert np.isclose(fingerprint(a_pinv), -0.3244041253699862)
assert rank == 161
fingerprint(a_pinv), rank

# ### qr


def _qr_canon(q, r):
    # Q/R are unique only up to the sign of each column; make R's diagonal
    # real-positive so numpy's and faer's conventions agree
    k = np.diag(r).size
    d = np.diag(r)
    d = d / np.abs(d)
    q = q.copy()
    q[:, :k] = q[:, :k] * d[np.newaxis, :]
    r = r.copy()
    r[:k, :] = r[:k, :] / d[:, np.newaxis]
    return q, r


# tall 1024x512; reduced (default) and complete share the thin Q and R
a = a_raw[:1024*512].copy().reshape(1024, 512)
q, r = np.linalg.qr(a)
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -3.816695392252389)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -245.99004994559718)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(1024, 512)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -3.816695392252389)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -245.99004994559718)
fingerprint(r)

# wide 512x1024
a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a)
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.9769304656721726)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 405.0109813743518)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.9769304656721726)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 405.0109813743518)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.9769304656721726)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 405.0109813743518)
fingerprint(r)

# ### lstsq

# the entry default rcond is eps * max(m, n); a vector right-hand side gives a
# scalar residual (the driver returns none when rank < min(m, n))
eps = np.finfo(np.float64).eps

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw[:1024].copy().reshape(1024)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(x), 0.6714195293796388)
assert np.isclose(fingerprint(s), 33.96933907104208)
assert np.isclose(fingerprint(residuals), 495.06381572902126)
assert rank == 512
fingerprint(x), fingerprint(s), fingerprint(residuals), rank

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw[:1024*3].copy().reshape(1024, 3)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(x), -2.417996890572109)
assert np.isclose(fingerprint(s), 33.96933907104208)
assert np.isclose(fingerprint(residuals), 544.9834618414125)
assert rank == 512
fingerprint(x), fingerprint(s), fingerprint(residuals), rank

a = a_raw[:512*1024].copy().reshape(512, 1024)
b = b_raw[:512].copy().reshape(512)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(x), 0.1316952775202293)
assert np.isclose(fingerprint(s), 32.27742168207754)
assert residuals.size == 0
assert rank == 512
fingerprint(x), fingerprint(s), fingerprint(residuals), rank

a = a_raw[:1024*512].copy().reshape(1024, 512)
a[:, 2] = a[:, 0] + a[:, 1]
b = b_raw[:1024].copy().reshape(1024)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(x), 0.6203201140848792)
assert np.isclose(fingerprint(s), 46.463897960241695)
assert np.isnan(residuals)
assert rank == 511
fingerprint(x), fingerprint(s), rank

# matrix b with a rank-deficient a: every residual column is NaN

a = a_raw[:1024*512].copy().reshape(1024, 512)
a[:, 2] = a[:, 0] + a[:, 1]
b = b_raw[:1024*3].copy().reshape(1024, 3)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(x), -2.475933070459636)
assert np.isclose(fingerprint(s), 46.463897960241695)
assert np.isnan(residuals).all()
assert rank == 511
fingerprint(x), fingerprint(s), rank, residuals
