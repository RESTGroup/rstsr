# # Driver tests in Python

import numpy as np
import scipy


def fingerprint(a):
    return np.dot(np.cos(np.arange(a.size)), np.asarray(a, order="C").ravel())


# Path of npy files in rstsr-test-manifest

root = "../../../../rstsr-test-manifest/resources/"

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


# ### eig

def test_eig():
    """Test eig and eigvals functions for general eigenvalue problems (complex)."""
    # Test 1: Basic eig with right eigenvectors (default) - 64x64
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vr = scipy.linalg.eig(a)
    assert np.isclose(fingerprint(np.abs(w)), 10.237632603014857)
    assert np.isclose(fingerprint(np.abs(vr)), -0.3765273892608302)

    # Test 2: eig with both left and right eigenvectors - 64x64
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vl, vr = scipy.linalg.eig(a, left=True, right=True)
    assert np.isclose(fingerprint(np.abs(w)), 10.237632603014857)
    assert np.isclose(fingerprint(np.abs(vl)), 5.802237057668709)
    assert np.isclose(fingerprint(np.abs(vr)), -0.3765273892608302)

    # Test 3: eig with left eigenvectors only - 64x64
    a = a_raw[:64*64].copy().reshape(64, 64)
    w, vl = scipy.linalg.eig(a, left=True, right=False)
    assert np.isclose(fingerprint(np.abs(w)), 10.237632603014857)
    assert np.isclose(fingerprint(np.abs(vl)), 5.802237057668709)

    # Test 4: eigvals (eigenvalues only) - 64x64
    a = a_raw[:64*64].copy().reshape(64, 64)
    w = scipy.linalg.eigvals(a)
    assert np.isclose(fingerprint(np.abs(w)), 10.237632603014857)

    # Test 5: Larger matrix - 512x512
    a = a_raw[:512*512].copy().reshape(512, 512)
    w, vr = scipy.linalg.eig(a)
    assert np.isclose(fingerprint(np.abs(w)), 5.178972257514008)
    assert np.isclose(fingerprint(np.abs(vr)), 4.707763397730002)

    # ===============================
    # Generalized eigenvalue problems
    # ===============================

    # Test 6: Generalized eigenvalue problem - 32x32
    a = a_raw[:32*32].copy().reshape(32, 32)
    b = b_raw[:32*32].copy().reshape(32, 32)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 9.593381733904854)
    # Verify eigenvalue equation: A @ vr[:,i] = w[i] * B @ vr[:,i]
    for i in range(min(3, len(w))):
        residual = np.linalg.norm(a @ vr[:, i] - w[i] * b @ vr[:, i])
        assert residual < 1e-6

    # Test 7: Generalized eigenvalue problem - 64x64
    a = a_raw[:64*64].copy().reshape(64, 64)
    b = b_raw[:64*64].copy().reshape(64, 64)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 15.169874441187222)

    # Test 8: Generalized eig with left and right eigenvectors - 32x32
    a = a_raw[:32*32].copy().reshape(32, 32)
    b = b_raw[:32*32].copy().reshape(32, 32)
    w, vl, vr = scipy.linalg.eig(a, b, left=True, right=True)
    assert np.isclose(fingerprint(np.abs(w)), 9.593381733904854)
    assert np.isclose(fingerprint(np.abs(vl)), 1.7160652746381113)
    assert np.isclose(fingerprint(np.abs(vr)), 1.2395098238300921)

    # Test 9: Generalized eigenvalue problem - 128x128
    a = a_raw[:128*128].copy().reshape(128, 128)
    b = b_raw[:128*128].copy().reshape(128, 128)
    w, vr = scipy.linalg.eig(a, b)
    assert np.isclose(fingerprint(np.abs(w)), 21.069841281001473)

    # Test 10: Simple 2x2 generalized eigenvalue problem (complex)
    a_simple = np.array([[1.0+0.5j, 2.0-0.3j], [3.0+0.2j, 4.0-0.1j]], dtype=np.complex64)
    b_simple = np.array([[2.0+0.1j, 1.0-0.2j], [1.0+0.3j, 2.0-0.4j]], dtype=np.complex64)
    w, vr = scipy.linalg.eig(a_simple, b_simple)
    assert np.isclose(fingerprint(np.abs(w)), 2.2248535308578066)
    # Verify eigenvalue equation
    for i in range(len(w)):
        residual = np.linalg.norm(a_simple @ vr[:, i] - w[i] * b_simple @ vr[:, i])
        assert residual < 1e-6


test_eig()


