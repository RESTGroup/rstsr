# # Driver tests in Python

import numpy as np
import scipy


def fingerprint(a):
    return np.dot(np.cos(np.arange(a.size)), np.asarray(a, order="C").ravel())


# Path of npy files in rstsr-test-manifest

root = "../../../../rstsr-test-manifest/resources/"

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

# ### dsyevr

# all eigenvalues (lower=True)
a = a_raw.copy().reshape(1024, 1024)
w, z, m, isuppz, _ = scipy.linalg.lapack.dsyevr(a, lower=True, range='A')
assert m == 1024
assert np.isclose(fingerprint(w), -71.4747209499407)
assert np.isclose(fingerprint(np.abs(z)), -9.903934930318247)
fingerprint(w), fingerprint(np.abs(z))

# all eigenvalues (lower=False)
a = a_raw.copy().reshape(1024, 1024)
w, z, m, isuppz, _ = scipy.linalg.lapack.dsyevr(a, lower=False, range='A')
assert m == 1024
assert np.isclose(fingerprint(w), -71.4902453763506)
assert np.isclose(fingerprint(np.abs(z)), 6.973792268793419)
fingerprint(w), fingerprint(np.abs(z))

# eigenvalues by index [0,99] (lower=True)
a = a_raw.copy().reshape(1024, 1024)
w, z, m, isuppz, _ = scipy.linalg.lapack.dsyevr(a, lower=True, range='I', il=1, iu=100)
assert m == 100
w_valid = w[:m]
assert np.isclose(fingerprint(w_valid), 7.172235948598356)
assert np.isclose(fingerprint(np.abs(z)), -4.561871974095643)
fingerprint(w_valid), fingerprint(np.abs(z))

# eigenvalues by index [500,599] (lower=True)
a = a_raw.copy().reshape(1024, 1024)
w, z, m, isuppz, _ = scipy.linalg.lapack.dsyevr(a, lower=True, range='I', il=501, iu=600)
assert m == 100
w_valid = w[:m]
assert np.isclose(fingerprint(w_valid), -8.66127659333348)
assert np.isclose(fingerprint(np.abs(z)), 4.814753342657916)
fingerprint(w_valid), fingerprint(np.abs(z))

# eigenvalues only (jobz='N')
a = a_raw.copy().reshape(1024, 1024)
w, z, m, isuppz, _ = scipy.linalg.lapack.dsyevr(a, lower=True, range='A', compute_v=False)
assert m == 1024
assert z.size == 0  # z should be empty when compute_v=False
assert np.isclose(fingerprint(w), -71.4747209499407)
fingerprint(w)

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

# ### dsygvx

# all eigenvalues (lower, should match SYGV)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='V', range='A', uplo='L')
assert m == 1024
assert np.isclose(fingerprint(w[:m]), -89.60433120129986)
assert np.isclose(fingerprint(np.abs(z[:, :m])), -5.243112559129755)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# all eigenvalues (upper)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='V', range='A', uplo='U')
assert m == 1024
assert np.isclose(fingerprint(w[:m]), -65.27252612342821)
assert np.isclose(fingerprint(np.abs(z[:, :m])), -7.084950485752325)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# eigenvalues by index [0,99] (lower)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='V', range='I', uplo='L', il=1, iu=100)
assert m == 100
assert np.isclose(fingerprint(w[:m]), -66.50936647104814)
assert np.isclose(fingerprint(np.abs(z[:, :m])), -2.4155996407215308)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# eigenvalues by index [500,599] (lower)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='V', range='I', uplo='L', il=501, iu=600)
assert m == 100
assert np.isclose(fingerprint(w[:m]), -0.31195670912953266)
assert np.isclose(fingerprint(np.abs(z[:, :m])), 0.36590319727369397)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# eigenvalues by value range [-10, 10] (lower)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='V', range='V', uplo='L', vl=-10.0, vu=10.0)
assert m == 990
assert np.isclose(fingerprint(w[:m]), -4.664359009761954)
assert np.isclose(fingerprint(np.abs(z[:, :m])), -3.898103312447499)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# eigenvalues only (jobz='N')
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=1, jobz='N', range='A', uplo='L')
assert m == 1024
assert z.size == 0
assert np.isclose(fingerprint(w[:m]), -89.60433120129974)
fingerprint(w[:m])

# itype=2, all eigenvalues (lower)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=2, jobz='V', range='A', uplo='L')
assert m == 1024
assert np.isclose(fingerprint(w[:m]), -2437.0943048614404)
assert np.isclose(fingerprint(np.abs(z[:, :m])), -4.108281604766399)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

# itype=3, all eigenvalues (lower)
a = a_raw.copy().reshape(1024, 1024)
b = b_raw.copy().reshape(1024, 1024)
w, z, m, ifail, _ = scipy.linalg.lapack.dsygvx(a, b, itype=3, jobz='V', range='A', uplo='L')
assert m == 1024
assert np.isclose(fingerprint(w[:m]), -2437.0943048614404)
assert np.isclose(fingerprint(np.abs(z[:, :m])), 30.756098926681165)
fingerprint(w[:m]), fingerprint(np.abs(z[:, :m]))

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


# ## qr driver tests

# ### dgeqrf

a = a_raw.copy()[:512*512].reshape(512, 512)
qr, tau, _, _ = scipy.linalg.lapack.dgeqrf(a)
assert np.isclose(fingerprint(np.abs(qr)), 104.85191447485762)
assert np.isclose(fingerprint(tau), 2.599625942247915)
fingerprint(np.abs(qr)), fingerprint(tau)

# ### dorgqr

a = a_raw.copy()[:512*512].reshape(512, 512)
qr, tau, _, _ = scipy.linalg.lapack.dgeqrf(a)
q, _, _ = scipy.linalg.lapack.dorgqr(qr, tau)
assert np.isclose(fingerprint(np.abs(q)), -1.6083348348387112)
fingerprint(np.abs(q))

# ### dgeqp3

a = a_raw.copy()[:512*512].reshape(512, 512)
qr, jpvt, tau, _, _ = scipy.linalg.lapack.dgeqp3(a)
assert np.isclose(fingerprint(np.abs(qr)), 82.18652098057946)
assert np.isclose(fingerprint(tau), 0.9651077849682483)
# jpvt is 1-indexed from LAPACK, verify it's a permutation
jpvt_sorted = np.sort(jpvt)
assert np.allclose(jpvt_sorted, np.arange(1, 513))
fingerprint(np.abs(qr)), fingerprint(tau), jpvt[:10]

# ### dormqr

a = a_raw.copy()[:512*512].reshape(512, 512)
c = b_raw.copy()[:512*256].reshape(512, 256)
qr, tau, _, _ = scipy.linalg.lapack.dgeqrf(a)
# Q^T * C: use lwork=-1 to query, then use optimal lwork
cq, work, info = scipy.linalg.lapack.dormqr('L', 'T', qr, tau, c, -1)
lwork = int(work[0])
cq, work, info = scipy.linalg.lapack.dormqr('L', 'T', qr, tau, c, lwork)
assert np.isclose(fingerprint(np.abs(cq)), 14.485921558590004)
fingerprint(np.abs(cq))


# ## eig driver tests

# ### dgeev

# Test with right eigenvectors only
a = a_raw.copy()[:512*512].reshape(512, 512)
wr, wi, vl, vr, _ = scipy.linalg.lapack.dgeev(a, compute_vl=False, compute_vr=True)
assert np.isclose(fingerprint(wr), 18.71988822379452)
assert np.isclose(fingerprint(wi), -44.508746930827016)
assert np.isclose(fingerprint(np.abs(vr)), 1.4038315907569003)
fingerprint(wr), fingerprint(wi), fingerprint(np.abs(vr))

# Test with left eigenvectors only
a = a_raw.copy()[:512*512].reshape(512, 512)
wr, wi, vl, vr, _ = scipy.linalg.lapack.dgeev(a, compute_vl=True, compute_vr=False)
assert np.isclose(fingerprint(wr), 18.71988822379452)
assert np.isclose(fingerprint(wi), -44.508746930827016)
assert np.isclose(fingerprint(np.abs(vl)), -3.014414428565583)
fingerprint(wr), fingerprint(wi), fingerprint(np.abs(vl))

# Test with both eigenvectors
a = a_raw.copy()[:512*512].reshape(512, 512)
wr, wi, vl, vr, _ = scipy.linalg.lapack.dgeev(a, compute_vl=True, compute_vr=True)
assert np.isclose(fingerprint(wr), 18.71988822379452)
assert np.isclose(fingerprint(wi), -44.508746930827016)
assert np.isclose(fingerprint(np.abs(vl)), -3.014414428565583)
assert np.isclose(fingerprint(np.abs(vr)), 1.4038315907569003)
fingerprint(wr), fingerprint(wi), fingerprint(np.abs(vl)), fingerprint(np.abs(vr))


