import numpy as np
import os

np.random.seed(42)

for dtype, dtype_name, dtype_name_complex in [
    (np.float32, "f32", "c32"),
    (np.float64, "f64", "c64"),
]:
    a = np.random.randn(4096*4096).astype(dtype)
    b = np.random.randn(4096*4096).astype(dtype)
    c = np.random.randn(4096*4096).astype(dtype)
    
    for name, arr in [
        (f"a-{dtype_name}.npy", a),
        (f"b-{dtype_name}.npy", b),
        (f"c-{dtype_name}.npy", c),
    ]:
        with open(os.path.join(os.path.dirname(__file__), name), "wb") as f:
            np.save(f, arr)

    a = np.random.randn(4096*4096).astype(dtype) + 1j * np.random.randn(4096*4096).astype(dtype)
    b = np.random.randn(4096*4096).astype(dtype) + 1j * np.random.randn(4096*4096).astype(dtype)
    c = np.random.randn(4096*4096).astype(dtype) + 1j * np.random.randn(4096*4096).astype(dtype)
    
    for name, arr in [
        (f"a-{dtype_name_complex}.npy", a),
        (f"b-{dtype_name_complex}.npy", b),
        (f"c-{dtype_name_complex}.npy", c),
    ]:
        with open(os.path.join(os.path.dirname(__file__), name), "wb") as f:
            np.save(f, arr)
