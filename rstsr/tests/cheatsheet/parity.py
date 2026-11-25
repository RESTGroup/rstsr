#!/usr/bin/env python3
"""NumPy/SciPy reference values for the RSTSR cheat sheet.

Run-and-print companion of `rstsr/src/docs/cheatsheet.md`,
`rstsr/src/docs/cheatsheet_linalg.md`, and the two integration tests
`rstsr/tests/cheatsheet_core.rs` / `cheatsheet_linalg.rs`.

Each block is labelled with `[<page>] <example_id>` where the id matches the
Rust test function of the same name; the three artifacts must stay in sync
(skill `api-cheatsheet-conventions`). This script never writes code — it only
prints reference values for a reviewer to compare against.

Usage:  python3 rstsr/tests/cheatsheet/parity.py
"""

import numpy as np

try:
    import scipy.linalg as sla

    HAVE_SCIPY = True
except ImportError:  # scipy is optional; the linalg blocks fall back to numpy
    HAVE_SCIPY = False

np.set_printoptions(precision=6, suppress=True)


def block(page, example_id, **values):
    print(f"# [{page}] {example_id}")
    for name, value in values.items():
        print(f"#   {name} = {value}")
    print()


def core_blocks():
    block("cheatsheet-core", "creation_special",
          zeros_sum=np.zeros((2, 3)).sum(),
          ones_sum=np.ones((2, 3)).sum(),
          full=np.full((2, 2), 7),
          eye_trace=np.eye(3).trace(),
          arange=np.arange(6.0),
          linspace=np.linspace(0.0, 1.0, 5))

    block("cheatsheet-core", "creation_from_lists",
          asarray=np.array([0, 1, 2, 3, 4, 5]),
          asarray_shaped=np.array([0, 1, 2, 3, 4, 5]).reshape(2, 3))

    m = np.arange(12).reshape(3, 4)
    block("cheatsheet-core", "indexing_basic",
          col=m[:, 3],
          block=m[0:2, 2:4],
          last=m[..., -1],
          newaxis_shape=m[:, None].shape,
          stepped=np.arange(6)[1:6:2],
          reversed=np.arange(6)[::-1])

    block("cheatsheet-core", "indexing_array",
          int_rows=m[[2, 0]][0],
          int_pairs=m[[2, 0], [1, 3]],
          int_scalar=m[[2, 0], 1],
          gather=m[:, [3, 1]][0],
          mask=m[np.array([[True, False, True, False],
                           [False, False, False, True],
                           [False, False, True, False]])],
          bool_shape=m[:, [True, False, True, False]].shape)

    block("cheatsheet-core", "manipulation_joining",
          concat=np.concatenate([np.arange(3), np.arange(3, 6)]),
          stack_shape=np.stack([np.arange(3), np.arange(3)]).shape,
          tile=np.tile([1, 2], (2, 2)),
          repeat=np.repeat([1, 2], 2),
          roll=np.roll(np.arange(3), 2),
          trace=np.arange(9).reshape(3, 3).trace())

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    block("cheatsheet-core", "operations_operators",
          add=a + b,
          scalar=a * 2,
          rem=a % b,
          matmul=np.arange(6).reshape(2, 3) @ np.arange(6).reshape(3, 2))

    x = np.array([1.0, 4.0, 9.0])
    block("cheatsheet-core", "operations_elementwise",
          sqrt=np.sqrt(x),
          clip=np.clip(x, 2, 5),
          greater=x > 3.0,
          where=np.where(x > 3.0, np.array([10.0, 20.0, 30.0]), x))

    r = np.arange(24.0).reshape(2, 3, 4)
    block("cheatsheet-core", "reduction_basic",
          sum=r.sum(),
          sum_axis=-1 and r.sum(axis=-1).shape,
          sum_axes=r.sum(axis=(0, 2)).shape,
          sum_keepdims=r.sum(axis=-1, keepdims=True).shape,
          mean=r.mean(),
          max=r.max(),
          min=r.min(),
          argmax=r.argmax(),
          count_nonzero=np.count_nonzero(r),
          cumulative_sum=np.arange(4.0).cumsum())

    s = np.array([3, 1, 3, 2, 1])
    block("cheatsheet-core", "sorting_and_set",
          sort=np.sort(s),
          argsort=np.argsort(s, kind="stable"),
          unique=np.unique(s),
          isin=np.isin(s, [1, 3]),
          searchsorted=np.searchsorted([1, 3, 5, 7], [2, 6]),
          diff=np.diff([1, 4, 9, 16]))

    block("cheatsheet-core", "compare_allclose",
          allclose=np.allclose([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 1e-12]))


def linalg_blocks():
    sym = np.array([[2.0, 1.0], [1.0, 2.0]])
    gen = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    square = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([9.0, 8.0])

    w, v = np.linalg.eigh(sym)
    block("cheatsheet-linalg", "eigh_standard", eigenvalues=w, eigenvectors=v)

    block("cheatsheet-linalg", "eigvalsh_standard", eigenvalues=np.linalg.eigvalsh(sym))

    comp = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
    w, v = np.linalg.eig(comp)
    block("cheatsheet-linalg", "eig_standard", eigenvalues=w, eigenvectors=v)

    block("cheatsheet-linalg", "eigvals_standard", eigenvalues=np.linalg.eigvals(comp))

    if HAVE_SCIPY:
        wg, vg = sla.eigh(sym, np.eye(2), lower=True)
        block("cheatsheet-linalg", "eigh_generalized", eigenvalues=wg, eigenvectors=vg)

    block("cheatsheet-linalg", "cholesky_lower", cholesky=np.linalg.cholesky(sym))

    q, r = np.linalg.qr(gen)
    block("cheatsheet-linalg", "qr_reduced", q=q, r=r)

    u, s, vt = np.linalg.svd(gen, full_matrices=False)
    block("cheatsheet-linalg", "svd_reconstruct", singular_values=s)

    block("cheatsheet-linalg", "solve_general_residual", solution=np.linalg.solve(square, b))

    tri = np.array([[2.0, 0.0], [1.0, 3.0]])
    if HAVE_SCIPY:
        sol = sla.solve_triangular(tri, np.array([4.0, 7.0]), lower=True)
    else:
        sol = np.linalg.solve(tri, np.array([4.0, 7.0]))
    block("cheatsheet-linalg", "solve_triangular_residual", solution=sol)

    block("cheatsheet-linalg", "inv_and_pinv",
          inverse=np.linalg.inv(square),
          pinv=np.linalg.pinv(square))

    sign, logabsdet = np.linalg.slogdet(square)
    block("cheatsheet-linalg", "det_and_slogdet",
          det=np.linalg.det(square),
          sign=sign,
          logabsdet=logabsdet)

    block("cheatsheet-linalg", "matrix_power_rank_norm",
          matrix_power=np.linalg.matrix_power(np.array([[1.0, 2.0], [3.0, 4.0]]), 2),
          rank=np.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]])),
          frobenius=np.linalg.norm(np.array([[1.0, 2.0], [3.0, 4.0]]), "fro"),
          vector_norm=np.linalg.norm(np.array([3.0, 4.0])))


def main():
    print(f"# numpy {np.__version__}, scipy {'present' if HAVE_SCIPY else 'MISSING'}\n")
    core_blocks()
    linalg_blocks()


if __name__ == "__main__":
    main()
