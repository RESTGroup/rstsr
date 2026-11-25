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
  `QrResult`, `SLogDetResult`, `PinvResult`, `EigResult`); each converts to a
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

- **`solve_symmetric`** has **no `DeviceFaer` implementation** (it is
  BLAS-only). Use `solve_general`, or an eigensolver-based approach.
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
