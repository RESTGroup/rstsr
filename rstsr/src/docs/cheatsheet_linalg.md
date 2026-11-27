# RSTSR ↔ NumPy/SciPy cheat sheet — linear algebra

The `rt::linalg` family, backed here by the **faer** device. It requires the
`linalg` feature (the BLAS-backed devices implement the same functions with more
overloads; that is out of scope for this page). For core tensor operations see
the [core cheat sheet](crate::cheatsheet).

The page assumes:

```rust,ignore
use rstsr::prelude::*;
let device = DeviceFaer::default(); // requires the `linalg` + `faer` features
```

- Calls take the tensor (or a tuple of arguments) as **one parameter**:
  `rt::linalg::eigh(&a)`, `rt::linalg::eigh((&a, Upper))`.
- **Row-major** throughout. The default triangular `uplo` follows the device
  order: [`Lower`] under `RowMajor` (`[`Upper`]` under `ColMajor`). The enum is
  [`FlagUpLo`], with `FlagUpLo::U` / `FlagUpLo::L` aliased as `Upper` / `Lower`.
- Multi-output functions return a named struct (`EighResult`, `SVDResult`,
  `QrResult`, `SLogDetResult`, `PinvResult`, `EigResult`, `LstsqResult`); each converts to a
  plain tuple with `.into()`.
- Every function has a fallible `_f` twin returning `rt::Result`.

## Hermitian eigenvalue problems

| NumPy / SciPy | RSTSR | Description |
|--|--|--|
| `np.linalg.eigh(a, UPLO='L')` | `rt::linalg::eigh(&a)` | Eigenvalues (ascending) and eigenvectors |
| `np.linalg.eigh(a, UPLO)` | `rt::linalg::eigh((&a, Upper))` | Explicit triangle |
| `scipy.linalg.eigh(a, b)` | `rt::linalg::eigh((&a, &b))` | Generalized `A x = λ B x` |
| `scipy.linalg.eigh(a, b, lower, type=2)` | `rt::linalg::eigh((&a, &b, Lower, 2))` | `A B x = λ x` |
| `scipy.linalg.eigh(a, b, lower, type=3)` | `rt::linalg::eigh((&a, &b, Lower, 3))` | `B A x = λ x` |
| `np.linalg.eigvalsh(a)` | `rt::linalg::eigvalsh(&a)` | Eigenvalues only |

`eigh(...).into()` yields `(eigenvalues, eigenvectors)`.

## General eigenvalue problems

| NumPy / SciPy | RSTSR | Description |
|--|--|--|
| `np.linalg.eig(a)` | `rt::linalg::eig(&a)` | Eigenvalues and (right) eigenvectors |
| `np.linalg.eigvals(a)` | `rt::linalg::eigvals(&a)` | Eigenvalues only |

`eig(...)` returns `EigResult { eigenvalues, left, right }` (complex dtype; `left`
is `None` here), and `.into()` yields `(w, v)`.

## Matrix decomposition

| NumPy / SciPy | RSTSR | Description |
|--|--|--|
| `np.linalg.cholesky(a)` | `rt::linalg::cholesky(&a)` | Lower-factor Cholesky |
| `np.linalg.cholesky(a, upper=True)` | `rt::linalg::cholesky((&a, Upper))` | Upper factor |
| `np.linalg.qr(a)` | `rt::linalg::qr(&a)` | Reduced QR (`Q`, `R`) |
| `scipy.linalg.qr(a, mode="complete")` | `rt::linalg::qr((&a, QrMode::Complete))` | Complete QR |
| `np.linalg.svd(a)` | `rt::linalg::svd(&a)` | Full SVD (`U`, `s`, `Vᵀ`) |
| `np.linalg.svd(a, full_matrices=False)` | `rt::linalg::svd((&a, false))` | Reduced SVD |
| `np.linalg.svdvals(a)` | `rt::linalg::svdvals(&a)` | Singular values only |

## Solving linear systems

These solve `A X = B`. There is no single `solve`; choose the function matching
the structure of `A`.

| NumPy / SciPy | RSTSR | Description |
|--|--|--|
| `scipy.linalg.solve(a, b)` | `rt::linalg::solve_general((&a, &b))` | General square `A` |
| `scipy.linalg.solve(a, b, overwrite_b=True)` | `rt::linalg::solve_general((&a, b.view_mut()))` | Solve in place into `b` |
| `scipy.linalg.solve_triangular(a, b, lower=True)` | `rt::linalg::solve_triangular((&a, &b, Lower))` | Lower-triangular `A` |
| `scipy.linalg.solve_triangular(a, b, lower=False)` | `rt::linalg::solve_triangular((&a, &b, Upper))` | Upper-triangular `A` |

Note `A` is the **first** argument (RSTSR solves `A x = B`), matching SciPy.

## Other linear-algebra operations

| NumPy / SciPy | RSTSR | Description |
|--|--|--|
| `np.linalg.inv(a)` | `rt::linalg::inv(&a)` | Matrix inverse |
| `np.linalg.pinv(a)` | `rt::linalg::pinv(&a)` | Moore-Penrose pseudoinverse |
| `scipy.linalg.pinv(a, atol, rtol)` | `rt::linalg::pinv((&a, atol, rtol))` | With a singular-value tolerance |
| `np.linalg.det(a)` | `rt::linalg::det(&a)` | Determinant |
| `np.linalg.slogdet(a)` | `rt::linalg::slogdet(&a)` | `(sign, logabsdet)` where `det = sign·e^logabsdet` |
| `np.linalg.matrix_power(a, 2)` | `rt::linalg::matrix_power((&a, 2))` | Integer matrix power |
| `np.linalg.matrix_rank(a)` | `rt::linalg::matrix_rank(&a)` | Rank (number of non-zero singular values) |
| `np.linalg.matrix_norm(a, ord)` | `rt::linalg::matrix_norm((&a, ord))` | Matrix norm (`'fro'`, `1`, `2`, `'nuc'`, ...) |
| `np.linalg.norm(a, ord)` (vector) | `rt::linalg::vector_norm((&a, ord))` | Vector norm (`1`, `2`, `'inf'`, ...) |

## `DeviceFaer` backend notes

- **`lstsq`** is **BLAS-only** (LAPACK `GELSD`): there is no `DeviceFaer`
  least-squares entry. `lstsq((&a, &b))` uses `rcond = eps·max(M, N)`;
  `lstsq((&a, &b, rcond))` sets it explicitly (`rcond <= 0` or `>= 1` is
  clamped to machine epsilon, where SciPy rejects negatives). The result is
  `LstsqResult { x, residuals, rank, s }` (destructurable as a 4-tuple):
  `x` is `(..., N, K)` (or `(..., N)` for a vector `b`), `rank` holds one
  value per matrix slice, and `residuals` is real — `(..., K)` for a tall
  full-rank slice, `NaN` where `rank < N`, empty where `M <= N` (the SciPy
  convention). `b` is never modified; an owned or mutable `a` whose batch
  needs no broadcast and whose layout is the call order is consumed in place
  and destroyed by the factorization (use `a.view()` to keep it).
- **`solve_symmetric`** has **no `DeviceFaer` implementation** (it is
  BLAS-only). Use `solve_general`, or an eigensolver-based approach.
- **`lu` / `lu_factor` / `lu_solve`** (SciPy-style LU, LAPACK `GETRF`) are
  **BLAS-only**: there are no `DeviceFaer` entries.
  - `lu(&a)` returns `LuResult { p, l, u }` (destructurable as a 3-tuple)
    with `A = L[P] @ U`: `p` holds the **0-based row indices** (SciPy's
    `p_indices=True`), `l` is `(..., M, K)` and `u` is `(..., K, N)` with
    `K = min(M, N)`. `lu((&a, true))` (SciPy's `permute_l=True`) returns
    `p = None` and `l = P @ L`, so that `A = L @ U`.
  - `lu_factor(&a)` returns `LuFactorResult { lu, piv }` (a 2-tuple): the
    packed factors (`L` strictly below the diagonal, `U` on and above it)
    and the 0-based pivot indices.
  - `lu_solve((&lu, &piv, &b))` solves `op(A) X = b` from them. SciPy's
    `lu_solve((lu, piv), b)` is `lu_solve(((&lu, &piv), &b))`, and the
    result of `lu_factor` may be passed as it is. An optional last
    [`FlagTrans`] (`N`, `T`, `C`) selects `op`. The explicit factors of `lu`
    work as well: `lu_solve((&factors, &b[, FlagTrans]))`.
  - A singular matrix is no error for `lu` / `lu_factor` (the factors are
    complete, with a zero on `U`'s diagonal); `lu_solve` then returns
    `inf` / `NaN`.
  - An in-order owned or mutable input of `lu_factor` is factored in place
    and destroyed, as is such an input of `lu`; an owned or mutable `b` of
    `lu_solve` receives the solution in place when its batch needs no
    broadcast. Pass `a.view()` / `&b` to keep the inputs.
- **`ldl`** (SciPy `ldl`, LAPACK `SYTRF` / `HETRF`) is **BLAS-only**: there
  is no `DeviceFaer` entry. `ldl(&a)` returns `LdlResult { lu, d, perm }`
  (destructurable as a 3-tuple) with `A = lu @ d @ luᴴ` and `lu[perm, :]`
  unit lower triangular; `d` is the dense block-diagonal matrix with 1×1 and
  2×2 blocks. `ldl((&a, Lower))` / `Upper` selects the triangle of `a` that
  is read and the outer factor (default `Lower`, on both device orders);
  `ldl((&a, false))` factorizes a complex **symmetric** matrix
  (`A = lu @ d @ luᵀ`). Only the selected triangle is read; a singular `d`
  (`info > 0`) is no error; an in-order owned or mutable input is consumed
  and destroyed (pass `a.view()` to keep it).
- **`cho_factor` / `cho_solve`** (SciPy `cho_factor` / `cho_solve`, LAPACK
  `POTRF` + two triangular solves) are **BLAS-only**: there are no
  `DeviceFaer` entries.
  - `cho_factor(&a)` returns `ChoFactorResult { c, uplo }` (destructurable
    as a 2-tuple): the factor sits in the `uplo` triangle, and `uplo`
    defaults to the device order's own triangle (`Lower` row-major, `Upper`
    column-major), so `cho_factor(&a).c` and `cholesky(&a)` carry the same
    triangle. The other triangle is zero for a read-only input (the SciPy
    default) and keeps the input's values when an in-order owned or mutable
    input was factored in place and destroyed (pass `a.view()` to keep it).
    A non-positive-definite matrix raises the `POTRF` error (`info = i`).
  - `cho_solve((&factors, &b))` solves `A x = b` from them. SciPy's
    `cho_solve((c, lower), b)` is `cho_solve(((c, Lower), b))` or
    `cho_solve(((c.view(), Lower), &b))`; the result of `cho_factor` may be
    passed as it is, as may the explicit factor of `cholesky`:
    `cho_solve(((l.view(), Lower), &b))`. Only the `uplo` triangle of the
    factor is read; a factor in either storage order is solved without
    copying it.
  - `b` is read-only (`&b`) for a fresh solution; an owned or mutable `b` is
    solved in place when its batch needs no broadcast.
- **`eigvalsh`** on faer handles the **standard** problem only; generalized
  eigenvalues go through `eigh((&a, &b, uplo, type))`.
- **`eig` / `eigvals`** on faer handle the **standard** problem only
  (`eig(&a)`, `eigvals(&a)`; right eigenvectors). The generalized form
  (`eig((&a, &b, side))`, `eigvals((&a, &b))`), the `EigSide` selections
  (`EigSide::Left` / `Both` / `None`) and the `Homogeneous` `(α, β)` spectrum are
  **BLAS-only**. The eigenvalues are always complex and unsorted on both
  backends (NumPy and SciPy return real arrays for a real spectrum); the
  generalized eigenvectors are rescaled to unit 2-norm (as SciPy does). On the
  BLAS devices the inputs are consumed as scratch: an in-order owned or
  mutable input is destroyed when its batch needs no broadcast.
- The builder structs (`EighArgs`, `SVDArgs`) are BLAS-only. On faer, use the
  tuple overloads shown above.
