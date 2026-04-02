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


# ### eig

def test_eig():
    """Test eig and eigvals functions for general eigenvalue problems."""
    # Test 1: Basic eig with right eigenvectors (default)
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vr = scipy.linalg.eig(a)
    assert np.isclose(fingerprint(np.abs(w)), 9.819876443763567)
    assert np.isclose(fingerprint(np.abs(vr)), -3.1323839657585237)

    # Test 2: eig with both left and right eigenvectors
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
    assert np.isclose(fingerprint(np.abs(w)), 9.819876443763567)
    assert np.isclose(fingerprint(np.abs(vl)), 0.30269389067674696)
    assert np.isclose(fingerprint(np.abs(vr)), -3.1323839657585237)

    # Test 3: eig with left eigenvectors only
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vl = scipy.linalg.eig(a, left=True, right=False)
    assert np.isclose(fingerprint(np.abs(w)), 9.819876443763567)
    assert np.isclose(fingerprint(np.abs(vl)), 0.30269389067674696)

    # Test 4: eigvals (eigenvalues only)
    a = a_raw[:64*64].copy().reshape(64, 64)
    w = scipy.linalg.eigvals(a)
    assert np.isclose(fingerprint(np.abs(w)), 9.819876443763567)

    # Test 5: Rotation matrix with complex eigenvalues
    theta = np.pi / 4
    a_rot = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    w, vr = scipy.linalg.eig(a_rot)
    assert np.isclose(fingerprint(np.abs(w)), 1.5403023058681398)
    assert np.isclose(fingerprint(np.abs(vr)), 0.09486754779484802)
    # Verify eigenvalue equation: A @ vr[:,i] = w[i] * vr[:,i]
    for i in range(len(w)):
        residual = np.linalg.norm(a_rot @ vr[:, i] - w[i] * vr[:, i])
        assert residual < 1e-10

    # Test 6: Matrix with mixed real/complex eigenvalues
    a_mixed = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(np.pi/3), -np.sin(np.pi/3)],
                        [0.0, np.sin(np.pi/3), np.cos(np.pi/3)]], dtype=np.float64)
    w, vr = scipy.linalg.eig(a_mixed)
    assert np.isclose(fingerprint(np.abs(w)), 1.1241554693209974)
    assert np.isclose(fingerprint(np.abs(vr)), -0.36634076382682534)

    # Test 7: Larger matrix (512x512)
    a = a_raw[:512*512].copy().reshape(512, 512)
    w, vr = scipy.linalg.eig(a)
    assert np.isclose(fingerprint(np.abs(w)), -0.3581013200396912)
    assert np.isclose(fingerprint(np.abs(vr)), 15.557771573864212)

    # ===============================
    # Generalized eigenvalue problems
    # ===============================

    # Test 8: Generalized eigenvalue problem (32x32)
    a = a_raw[:32*32].copy().reshape(32, 32)
    b = b_raw[:32*32].copy().reshape(32, 32)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 145.30492158178672)
    # Verify eigenvalue equation: A @ vr[:,i] = w[i] * B @ vr[:,i]
    for i in range(min(3, len(w))):
        residual = np.linalg.norm(a @ vr[:, i] - w[i] * b @ vr[:, i])
        assert residual < 1e-10

    # Test 9: Generalized eigenvalue problem (64x64)
    a = a_raw[:64*64].copy().reshape(64, 64)
    b = b_raw[:64*64].copy().reshape(64, 64)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 166.30282160221182)

    # Test 10: Generalized eig with left and right eigenvectors (32x32)
    a = a_raw[:32*32].copy().reshape(32, 32)
    b = b_raw[:32*32].copy().reshape(32, 32)
    w, vl, vr = scipy.linalg.eig(a, b, left=True, right=True)
    assert np.isclose(fingerprint(np.abs(w)), 145.30492158178672)
    assert np.isclose(fingerprint(np.abs(vl)), 3.1071697883122544)
    assert np.isclose(fingerprint(np.abs(vr)), 1.5806610713013591)

    # Test 11: Generalized eig with left eigenvectors only (32x32)
    a = a_raw[:32*32].copy().reshape(32, 32)
    b = b_raw[:32*32].copy().reshape(32, 32)
    w, vl = scipy.linalg.eig(a, b, left=True, right=False)
    assert np.isclose(fingerprint(np.abs(w)), 145.30492158178672)

    # Test 12: Generalized eigenvalue problem (128x128)
    a = a_raw[:128*128].copy().reshape(128, 128)
    b = b_raw[:128*128].copy().reshape(128, 128)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 23.2127389400861)

    # Test 13: Simple 2x2 generalized eigenvalue problem
    a_simple = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b_simple = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    w, vr = scipy.linalg.eig(a_simple, b_simple)
    assert np.isclose(fingerprint(np.abs(w)), 1.4139379450696126)
    # Verify eigenvalue equation
    for i in range(len(w)):
        residual = np.linalg.norm(a_simple @ vr[:, i] - w[i] * b_simple @ vr[:, i])
        assert residual < 1e-10


test_eig()


