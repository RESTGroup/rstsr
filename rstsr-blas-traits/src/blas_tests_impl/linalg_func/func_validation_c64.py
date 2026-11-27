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

a_raw = np.load(f"{root}/a-c64.npy")
b_raw = np.load(f"{root}/b-c64.npy")

assert np.isclose(fingerprint(a_raw), 191.28900005102915+217.50386287824938j)
assert np.isclose(fingerprint(b_raw), 267.6279081341384-641.4397224458443j)

# ## tests

# ### cholesky

b = b_raw.copy().reshape(1024, 1024)
c = np.linalg.cholesky(b)
assert np.isclose(fingerprint(c), 62.89494065393874-73.47055443374522j)
fingerprint(c)

b = b_raw.copy().reshape(1024, 1024)
c = scipy.linalg.cholesky(b, lower=False)
assert np.isclose(fingerprint(c), 13.720509103165073-1.8066465348490963j)
fingerprint(c)

# ### eigh

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eigh(a)
assert np.isclose(fingerprint(w), -100.79793355894122)
assert np.isclose(fingerprint(np.abs(v)), -7.450761195788254)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eigh(a, UPLO="U")
assert np.isclose(fingerprint(w), -103.99103522434956)
assert np.isclose(fingerprint(np.abs(v)), -12.184946930165328)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, lower=True)
assert np.isclose(fingerprint(w), -97.43376763322635)
assert np.isclose(fingerprint(np.abs(v)), -4.3181177983574255)
fingerprint(w), fingerprint(np.abs(v))

a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, lower=False, type=3)
assert np.isclose(fingerprint(w), -4656.824753078057)
assert np.isclose(fingerprint(np.abs(v)), -0.15861903557045487)
fingerprint(w), fingerprint(np.abs(v))

# ### Tests of eigh (itype, lower)

# 1, lower
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=1, lower=True)
assert np.isclose(fingerprint(w), -97.43376763322635)
assert np.isclose(fingerprint(np.abs(v)), -4.3181177983574255)
fingerprint(w), fingerprint(np.abs(v))

# 1, upper
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=1, lower=False)
assert np.isclose(fingerprint(w), -54.81859256480441)
assert np.isclose(fingerprint(np.abs(v)), -1.4841788446757156)
fingerprint(w), fingerprint(np.abs(v))

# 2, lower
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=2, lower=True)
assert np.isclose(fingerprint(w), -4967.627482507203)
assert np.isclose(fingerprint(np.abs(v)), 5.541034627252399)
fingerprint(w), fingerprint(np.abs(v))

# 2, upper
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=2, lower=False)
assert np.isclose(fingerprint(w), -4656.824753078057)
assert np.isclose(fingerprint(np.abs(v)), 1.0609263552377188)
fingerprint(w), fingerprint(np.abs(v))

# 3, lower
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=3, lower=True)
assert np.isclose(fingerprint(w), -4967.627482507203)
assert np.isclose(fingerprint(np.abs(v)), 118.76501084045631)
fingerprint(w), fingerprint(np.abs(v))

# 3, upper
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, v = scipy.linalg.eigh(a, b, type=3, lower=False)
assert np.isclose(fingerprint(w), -4656.824753078057)
assert np.isclose(fingerprint(np.abs(v)), -0.15861903557045487)
fingerprint(w), fingerprint(np.abs(v))

# ### eig

a = a_raw.copy().reshape(1024, 1024)
w, v = np.linalg.eig(a)
# a non-Hermitian spectrum is unordered; sort to a canonical order so the
# reference is backend-independent (np.sort / np.argsort on complex sort
# lexicographically by real then imaginary)
order = np.argsort(w)
assert np.isclose(fingerprint(np.sort(w)), -53.51484095091083-871.7616347408099j)
fingerprint(np.sort(w))
# eigenvector columns reordered to the same canonical order, then the
# elementwise absolute value (phase- and sign-invariant) before fingerprinting
assert np.isclose(fingerprint(np.abs(v[:, order])), 4.93240163375621)
fingerprint(np.abs(v[:, order]))

# ### eigvals

a = a_raw.copy().reshape(1024, 1024)
w = np.linalg.eigvals(a)
assert np.isclose(fingerprint(np.sort(w)), -53.51484095091083-871.7616347408099j)
fingerprint(np.sort(w))

# ### eig (left / both)

a = a_raw.copy().reshape(1024, 1024)
w, vl = scipy.linalg.eig(a, left=True, right=False)
order = np.argsort(w)
assert np.isclose(fingerprint(np.sort(w)), -53.51484095091083-871.7616347408099j)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -5.456470679201179)
fingerprint(np.sort(w)), fingerprint(np.abs(vl[:, order]))

a = a_raw.copy().reshape(1024, 1024)
w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
order = np.argsort(w)
assert np.isclose(fingerprint(np.abs(vl[:, order])), -5.456470679201179)
assert np.isclose(fingerprint(np.abs(vr[:, order])), 4.932401633755345)
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
b = b_raw[:256*256].copy().reshape(256, 256)

w, vr = scipy.linalg.eig(a, b)
assert np.isclose(_fp_spectrum(w)[0], -38.302519430567614)
assert np.isclose(_fp_spectrum(w)[1], -73.72687717058412)
assert np.isclose(_fp_vecs(vr), -1.7966116235924798)
_fp_spectrum(w), _fp_vecs(vr)

a = a_raw[:256*256].copy().reshape(256, 256)
b = b_raw[:256*256].copy().reshape(256, 256)

w, vl, vr = scipy.linalg.eig(a, b, left=True, right=True)
assert np.isclose(_fp_vecs(vl), -1.4726483066580205)
assert np.isclose(_fp_vecs(vr), -1.7966116235924798)
_fp_vecs(vl), _fp_vecs(vr)

# ### eigvals (generalized)

a = a_raw[:256*256].copy().reshape(256, 256)
b = b_raw[:256*256].copy().reshape(256, 256)

w = scipy.linalg.eigvals(a, b)
assert np.isclose(_fp_spectrum(w)[0], -38.302519430567614)
assert np.isclose(_fp_spectrum(w)[1], -73.72687717058412)
_fp_spectrum(w)

# ### inv

a = a_raw.copy().reshape(1024, 1024)
a_inv = np.linalg.inv(a)
assert np.isclose(fingerprint(a_inv), -11.836382515156183+8.250167298349842j)
fingerprint(a_inv)

# ### solve_general

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = np.linalg.solve(a, b)
assert np.isclose(fingerprint(x), 404.1900761036138-258.5602505551204j)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024].copy().reshape(1024)
x = np.linalg.solve(a, b)
assert np.isclose(fingerprint(x), -15.070310793269726-1.987917054716041j)
fingerprint(x)

# ### sovle_symmetric

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="sym", lower=True)
assert np.isclose(fingerprint(x), 401.05642312535775-805.8028453625365j)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="sym", lower=False)
assert np.isclose(fingerprint(x), 141.70122084637046-829.609691493499j)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="her", lower=True)
assert np.isclose(fingerprint(x), -1053.7242100144504-559.2846004618166j)
fingerprint(x)

a = a_raw.copy().reshape(1024, 1024)
b = b_raw[:1024*512].copy().reshape(1024, 512)
x = scipy.linalg.solve(a, b, assume_a="her", lower=False)
assert np.isclose(fingerprint(x), 674.2725854112028-68.55236080351166j)
fingerprint(x)

# ### sovle_triangular

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw.copy().reshape(1024, 1024)
x = scipy.linalg.solve(b, a, assume_a="lower triangular")
assert np.isclose(fingerprint(x), -8.433708003916948+20.578272827017052j)
fingerprint(x)

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw.copy().reshape(1024, 1024)
x = scipy.linalg.solve(b, a, assume_a="upper triangular")
assert np.isclose(fingerprint(x), 0.1778922244846507+11.42463765128442j)
fingerprint(x)

# ### slogdot

a = a_raw.copy().reshape(1024, 1024)
sgn, logabsdet = np.linalg.slogdet(a)
assert np.isclose(sgn, -0.44606842323663365+0.8949988613351316j)
assert np.isclose(logabsdet, 3393.6720579594585)
sgn, logabsdet

# ### det

a = a_raw[:25].copy().reshape(5, 5)
det = np.linalg.det(a)
assert np.isclose(det, -24.808965756481086+11.800248863799464j)
det

# ### svd

a = a_raw[:1024*512].copy().reshape(1024, 512)
(u, s, vt) = scipy.linalg.svd(a)
assert np.isclose(fingerprint(np.abs(u)), -15.44133470545584)
assert np.isclose(fingerprint(s), 46.60343405921802)
assert np.isclose(fingerprint(np.abs(vt)), 2.1605324161714172)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

a = a_raw[:1024*512].copy().reshape(1024, 512)
(u, s, vt) = scipy.linalg.svd(a, full_matrices=False)
assert np.isclose(fingerprint(np.abs(u)), -1.9516528722381659)
assert np.isclose(fingerprint(s), 46.60343405921802)
assert np.isclose(fingerprint(np.abs(vt)), 2.1605324161714172)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

a = a_raw[:1024*512].copy().reshape(1024, 512)
s = scipy.linalg.svd(a, compute_uv=False)
assert np.isclose(fingerprint(s), 46.60343405921802)
fingerprint(s)

a = a_raw[:1024*512].copy().reshape(512, 1024)
(u, s, vt) = scipy.linalg.svd(a, full_matrices=False)
assert np.isclose(fingerprint(np.abs(u)), 4.636614351700778)
assert np.isclose(fingerprint(s), 47.599274835886646)
assert np.isclose(fingerprint(np.abs(vt)), 1.4497879458575658)
fingerprint(np.abs(u)), fingerprint(s), fingerprint(np.abs(vt))

# ### pinv

a = a_raw[:1024*512].copy().reshape(1024, 512)
a_pinv, rank = scipy.linalg.pinv(a, return_rank=True, atol=20, rtol=0.3)
assert np.isclose(fingerprint(a_pinv), -0.03454885412959018-0.023651876085623254j)
assert rank == 240
fingerprint(a_pinv), rank

a = a_raw[:1024*512].copy().reshape(512, 1024)
a_pinv, rank = scipy.linalg.pinv(a, return_rank=True, atol=20, rtol=0.3)
assert np.isclose(fingerprint(a_pinv), -0.2814806469687325-0.15198888300458474j)
assert rank == 240
fingerprint(a_pinv), rank

# ### qr


def _qr_canon(q, r):
    # Q/R are unique only up to the unit phase of each column; make R's diagonal
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
assert np.isclose(fingerprint(np.abs(q[:, :512])), -10.064742559299738)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 0.24039511237941724+239.69822983660995j)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(1024, 512)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -10.064742559299738)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 0.24039511237941724+239.69822983660995j)
fingerprint(r)

# wide 512x1024
a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a)
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.7203599314373099)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -47.17594962775703+808.8519384383662j)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.7203599314373099)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -47.17594962775703+808.8519384383662j)
fingerprint(r)


def _qr_canon(q, r):
    # Q/R are unique only up to a unit factor per column; make R's diagonal
    # real-positive so numpy's and rstsr's conventions agree
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
assert np.isclose(fingerprint(np.abs(q[:, :512])), -10.064742559299749)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 0.2403951123793533+239.69822983660924j)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(1024, 512)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -10.064742559299749)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), 0.2403951123793533+239.69822983660924j)
fingerprint(r)

# wide 512x1024
a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a)
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.7203599314373182)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -47.17594962775998+808.8519384383669j)
fingerprint(r)

a = a_raw[:1024*512].copy().reshape(512, 1024)
q, r = np.linalg.qr(a, mode="complete")
q, r = _qr_canon(q, r)
assert np.isclose(fingerprint(np.abs(q[:, :512])), -0.7203599314373182)
fingerprint(np.abs(q[:, :512]))
assert np.isclose(fingerprint(r), -47.17594962775998+808.8519384383669j)
fingerprint(r)

# ### lstsq

# the entry default rcond is eps * max(m, n); the driver returns no residuals
# for wide matrices
eps = np.finfo(np.float64).eps

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw[:1024].copy().reshape(1024, 1)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 0.46834637017620495)
assert np.isclose(fingerprint(s), 46.603434059217975)
assert np.isclose(fingerprint(residuals), 1062.17917639187)
assert rank == 512
fingerprint(np.abs(x)), fingerprint(s), fingerprint(residuals), rank

a = a_raw[:1024*512].copy().reshape(1024, 512)
b = b_raw[:1024*2].copy().reshape(1024, 2)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 0.10730347392277714)
assert np.isclose(fingerprint(s), 46.603434059217975)
assert np.isclose(fingerprint(residuals), 1613.436740523908)
assert rank == 512
fingerprint(np.abs(x)), fingerprint(s), fingerprint(residuals), rank

a = a_raw[:512*1024].copy().reshape(512, 1024)
b = b_raw[:512].copy().reshape(512, 1)
x, residuals, rank, s = scipy.linalg.lstsq(a, b, cond=eps*1024, lapack_driver="gelsd", check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 0.7685570101997597)
assert np.isclose(fingerprint(s), 47.59927483588676)
assert residuals.size == 0
assert rank == 512
fingerprint(np.abs(x)), fingerprint(s), fingerprint(residuals), rank

# ### lu

a = a_raw[:1024*1024].copy().reshape(1024, 1024)
p, l, u = scipy.linalg.lu(a, p_indices=True, check_finite=False)
assert np.isclose(fingerprint(p), 10021.799952379515)
assert np.isclose(fingerprint(np.abs(l)), 101.3884312729007)
assert np.isclose(fingerprint(np.abs(u)), -2781.9082366770954)
fingerprint(p), fingerprint(np.abs(l)), fingerprint(np.abs(u))

a = a_raw[:1024*512].copy().reshape(1024, 512)
p, l, u = scipy.linalg.lu(a, p_indices=True, check_finite=False)
assert np.isclose(fingerprint(p), -358.16906335896806)
assert np.isclose(fingerprint(np.abs(l)), 93.90580050538996)
assert np.isclose(fingerprint(np.abs(u)), -350.42517036956383)
fingerprint(p), fingerprint(np.abs(l)), fingerprint(np.abs(u))

# permute_l=True: (P L, U) without P

a = a_raw[:1024*1024].copy().reshape(1024, 1024)
pl, u = scipy.linalg.lu(a, permute_l=True, check_finite=False)
assert np.isclose(fingerprint(np.abs(pl)), -78.03219179967243)
assert np.isclose(fingerprint(np.abs(u)), -2781.9082366770954)
fingerprint(np.abs(pl)), fingerprint(np.abs(u))

# ### lu_factor

a = a_raw[:1024*1024].copy().reshape(1024, 1024)
lu, piv = scipy.linalg.lu_factor(a, check_finite=False)
assert np.isclose(fingerprint(np.abs(lu)), -2680.7252882699167)
assert np.isclose(fingerprint(piv.astype(np.float64)), 1553.3051188216255)
fingerprint(np.abs(lu)), fingerprint(piv.astype(np.float64))

a = a_raw[:1024*512].copy().reshape(1024, 512)
lu, piv = scipy.linalg.lu_factor(a, check_finite=False)
assert np.isclose(fingerprint(np.abs(lu)), -256.51823408143457)
assert np.isclose(fingerprint(piv.astype(np.float64)), -562.1343902757835)
fingerprint(np.abs(lu)), fingerprint(piv.astype(np.float64))

# ### lu_solve

lu, piv = scipy.linalg.lu_factor(a_raw[:1024*1024].copy().reshape(1024, 1024), check_finite=False)

b = b_raw[:1024].copy()
x = scipy.linalg.lu_solve((lu, piv), b, trans=0, check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 22.15319665971149)
fingerprint(np.abs(x))

x = scipy.linalg.lu_solve((lu, piv), b, trans=2, check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 1.9240075469181859)
fingerprint(np.abs(x))

# ### ldl

ac = a_raw[:1024*1024].copy().reshape(1024, 1024)
ah, asym = ac + ac.conj().T, ac + ac.T
lu, d, perm = scipy.linalg.ldl(ah, check_finite=False)
assert np.isclose(fingerprint(np.abs(lu)), -40.081477868486445)
assert np.isclose(fingerprint(np.abs(d)), -755.721723672091912)
assert np.isclose(fingerprint(perm.astype(np.float64)), -5295.552240978868213)
fingerprint(np.abs(lu)), fingerprint(np.abs(d)), fingerprint(perm.astype(np.float64))

lu, d, perm = scipy.linalg.ldl(ah, lower=False, check_finite=False)
assert np.isclose(fingerprint(np.abs(lu)), 51.808720927208192)
assert np.isclose(fingerprint(np.abs(d)), -180.086817996708049)
assert np.isclose(fingerprint(perm.astype(np.float64)), 2583.421176397792806)
fingerprint(np.abs(lu)), fingerprint(np.abs(d)), fingerprint(perm.astype(np.float64))

lu, d, perm = scipy.linalg.ldl(asym, hermitian=False, check_finite=False)
assert np.isclose(fingerprint(np.abs(lu)), 33.914606804494227)
assert np.isclose(fingerprint(np.abs(d)), 184.865319971126183)
assert np.isclose(fingerprint(perm.astype(np.float64)), 1695.094480873951852)
fingerprint(np.abs(lu)), fingerprint(np.abs(d)), fingerprint(perm.astype(np.float64))

# ### cho_factor

b = b_raw[:1024*1024].copy().reshape(1024, 1024)
c, lower = scipy.linalg.cho_factor(b, lower=True, check_finite=False)
assert np.isclose(fingerprint(c), 62.894940653939656-73.4705544337455j)
fingerprint(c)

c, lower = scipy.linalg.cho_factor(b, lower=False, check_finite=False)
assert np.isclose(fingerprint(c), 13.720509103164034-1.8066465348487615j)
fingerprint(c)

# ### cho_solve

c, _ = scipy.linalg.cho_factor(b, lower=True, check_finite=False)
b_rhs = a_raw[:1024].copy()
x = scipy.linalg.cho_solve((c, True), b_rhs, check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 5.093725636509232)
fingerprint(np.abs(x))

c, _ = scipy.linalg.cho_factor(b, lower=False, check_finite=False)
b_rhs = a_raw[:1024].copy()
x = scipy.linalg.cho_solve((c, False), b_rhs, check_finite=False)
assert np.isclose(fingerprint(np.abs(x)), 5.093725636509143)
fingerprint(np.abs(x))

# ### matrix_norm

a = a_raw.copy().reshape(1024, 1024)
n = np.asarray(np.linalg.norm(a, ord="fro"))
assert np.isclose(fingerprint(n), 1449.1955776423124)
print("c64 matrix_norm fro ", fingerprint(n))
n = np.asarray(np.linalg.norm(a, ord=1))
assert np.isclose(fingerprint(n), 1359.9335864360062)
print("c64 matrix_norm 1   ", fingerprint(n))
n = np.asarray(np.linalg.norm(a, ord=-1))
assert np.isclose(fingerprint(n), 1222.892551470915)
print("c64 matrix_norm -1  ", fingerprint(n))
n = np.asarray(np.linalg.norm(a, ord=np.inf))
assert np.isclose(fingerprint(n), 1365.5480489523034)
print("c64 matrix_norm inf ", fingerprint(n))
n = np.asarray(np.linalg.norm(a, ord=-np.inf))
assert np.isclose(fingerprint(n), 1219.794834983426)
print("c64 matrix_norm -inf", fingerprint(n))

a256 = a_raw.copy().reshape(-1)[: 256 * 256].reshape(256, 256)
n = np.asarray(np.linalg.norm(a256, ord="nuc"))
assert np.isclose(fingerprint(n), 4942.898431230364)
print("c64 matrix_norm nuc ", fingerprint(n))
n = np.asarray(np.linalg.norm(a256, ord=2))
assert np.isclose(fingerprint(n), 44.84157812023554)
print("c64 matrix_norm 2   ", fingerprint(n))
n = np.asarray(np.linalg.norm(a256, ord=-2))
assert np.isclose(fingerprint(n), 0.023430892892404798)
print("c64 matrix_norm -2  ", fingerprint(n))

# ### vector_norm

v = a_raw.copy().reshape(-1)
n = np.asarray(np.linalg.norm(v, ord=2))
assert np.isclose(fingerprint(n), 1449.1955776423124)
print("c64 vector_norm 2   ", fingerprint(n))
n = np.asarray(np.linalg.norm(v, ord=1))
assert np.isclose(fingerprint(n), 1315110.649779901)
print("c64 vector_norm 1   ", fingerprint(n))
n = np.asarray(np.linalg.norm(v, ord=np.inf))
assert np.isclose(fingerprint(n), 5.418092204281868)
print("c64 vector_norm inf ", fingerprint(n))
n = np.asarray(np.linalg.norm(v, ord=-np.inf))
assert np.isclose(fingerprint(n), 0.0008315180802683977)
print("c64 vector_norm -inf", fingerprint(n))
n = np.asarray(np.linalg.norm(v, ord=3))
assert np.isclose(fingerprint(n), 158.08987792483444)
print("c64 vector_norm 3   ", fingerprint(n))

# ### matrix_rank

a512 = a_raw.copy().reshape(-1)[: 1024 * 512].reshape(1024, 512)
r = np.linalg.matrix_rank(a512)
assert r == 512
print("c64 matrix_rank 1024x512 rank:", r)

# ### matrix_power

a4 = a_raw.copy().reshape(-1)[:16].reshape(4, 4)
p = np.asarray(np.linalg.matrix_power(a4, 0))
assert np.isclose(fingerprint(p), (-0.3150972564720475+0j))
print(f"c64 matrix_power 4x4 n=0  ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, 1))
assert np.isclose(fingerprint(p), (-0.5537625095088878-3.0610178896100395j))
print(f"c64 matrix_power 4x4 n=1  ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, 2))
assert np.isclose(fingerprint(p), (-3.3981320140425257+0.16690991094120955j))
print(f"c64 matrix_power 4x4 n=2  ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, 3))
assert np.isclose(fingerprint(p), (-14.635121297638918-1.0839909116245399j))
print(f"c64 matrix_power 4x4 n=3  ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, 5))
assert np.isclose(fingerprint(p), (-7.659626934528033+29.380294417607516j))
print(f"c64 matrix_power 4x4 n=5  ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, -1))
assert np.isclose(fingerprint(p), (0.23944130706023403+0.38201618793714387j))
print(f"c64 matrix_power 4x4 n=-1 ", fingerprint(p))
p = np.asarray(np.linalg.matrix_power(a4, -2))
assert np.isclose(fingerprint(p), (-0.40283871318247577+0.22747768835164975j))
print(f"c64 matrix_power 4x4 n=-2 ", fingerprint(p))

a = a_raw.copy().reshape(1024, 1024)
p = np.eye(1024, dtype=complex)
assert np.isclose(fingerprint(p), (0.20548286571538066+0j))
print("c64 matrix_power 1024 n=0 ", fingerprint(p))
p = np.asarray(a)
assert np.isclose(fingerprint(p), (191.28900005103077+217.5038628782485j))
print("c64 matrix_power 1024 n=1 ", fingerprint(p))
p = a @ a
assert np.isclose(fingerprint(p), (1565.4184664848617-66445.91164508468j))
print("c64 matrix_power 1024 n=2 ", fingerprint(p))
p = np.linalg.inv(a)
assert np.isclose(fingerprint(p), (-11.836382515155854+8.250167298349126j))
print("c64 matrix_power 1024 n=-1", fingerprint(p))
