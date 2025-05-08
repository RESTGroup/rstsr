import numpy as np
import os

np.random.seed(42)

for dtype, dtype_name, dtype_name_complex in [
    (np.float32, "f32", "c32"),
    (np.float64, "f64", "c64"),
]:
    np.random.seed(42)
    # a is random tensor
    a = np.random.randn(1024*1024).astype(dtype)
    # b is special tensor of positive-definite
    b = np.random.randn(1024*1024).astype(dtype)
    w, v = np.linalg.eigh(b.reshape(1024, 1024))
    b = (v @ np.diag(np.abs(w)) @ v.T).reshape(-1)
    
    for name, arr in [
        (f"a-{dtype_name}.npy", a),
        (f"b-{dtype_name}.npy", b),
    ]:
        with open(os.path.join(os.path.dirname(__file__), name), "wb") as f:
            np.save(f, arr)

    np.random.seed(42)
    a = np.random.randn(1024*1024).astype(dtype) + 1j * np.random.randn(1024*1024).astype(dtype)
    # b is special tensor of positive-definite
    b = np.random.randn(1024*1024).astype(dtype) + 1j * np.random.randn(1024*1024).astype(dtype)
    w, v = np.linalg.eigh(b.reshape(1024, 1024))
    b = (v @ np.diag(np.abs(w)) @ v.T).reshape(-1)
    
    for name, arr in [
        (f"a-{dtype_name_complex}.npy", a),
        (f"b-{dtype_name_complex}.npy", b),
    ]:
        with open(os.path.join(os.path.dirname(__file__), name), "wb") as f:
            np.save(f, arr)
