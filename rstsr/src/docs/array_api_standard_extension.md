## Linear Algebra (extensions)

| status | implementation | Python API | description |
|-|-|-|-|
| Y | `rt::linalg::cholesky` | [`cholesky`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cholesky.html) | Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix `x`. |
| Y | `rt::cross` | [`cross`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cross.html) | Returns the cross product of 3-element vectors. |
| Y | `rt::linalg::det` | [`det`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.det.html) | Returns the determinant of a square matrix (or a stack of square matrices) `x`. |
| Y | `rt::diagonal` | [`diagonal`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.diagonal.html) | Returns the specified diagonals of a matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::eigh` | [`eigh`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigh.html) | Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::eigvalsh` | [`eigvalsh`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigvalsh.html) | Returns the eigenvalues of a complex Hermitian or real symmetric matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::inv` | [`inv`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.inv.html) | Returns the multiplicative inverse of a square matrix (or a stack of square matrices) `x`. |
| Y | `rt::matmul` | [`matmul`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matmul.html) | Computes the matrix product. |
| Y | `rt::linalg::matrix_norm` | [`matrix_norm`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_norm.html) | Computes the matrix norm of a matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::matrix_power` | [`matrix_power`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_power.html) | Raises a square matrix (or a stack of square matrices) `x` to an integer power `n`. |
| Y | `rt::linalg::matrix_rank` | [`matrix_rank`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_rank.html) | Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices). |
| Y | `rt::matrix_transpose` | [`matrix_transpose`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_transpose.html) | Transposes a matrix (or a stack of matrices) x. |
| Y | `rt::outer` | [`outer`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.outer.html) | Returns the outer product of two vectors `x1` and `x2`. |
| Y | `rt::linalg::pinv` | [`pinv`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.pinv.html) | Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::qr` | [`qr`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.qr.html) | Returns the QR decomposition of a full column rank matrix (or a stack of matrices). |
| Y | `rt::linalg::slogdet` | [`slogdet`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.slogdet.html) | Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) `x`. |
| Y | `rt::linalg::solve_general` | [`solve`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.solve.html) | Returns the solution of a square system of linear equations with a unique solution. |
| Y | `rt::linalg::svd` | [`svd`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svd.html) | Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) `x`. |
| Y | `rt::linalg::svdvals` | [`svdvals`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svdvals.html) | Returns the singular values of a matrix (or a stack of matrices) `x`. |
| Y | `rt::tensordot` | [`tensordot`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.tensordot.html) | Returns a tensor contraction of x1 and x2 over specific axes. |
| Y | `rt::trace` | [`trace`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.trace.html) | Returns the sum along the specified diagonals of a matrix (or a stack of matrices) `x`. |
| Y | `rt::vecdot` | [`vecdot`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.vecdot.html) | Computes the (vector) dot product of two arrays. |
| Y | `rt::linalg::vector_norm` | [`vector_norm`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.vector_norm.html) | Computes the vector norm of a vector (or batch of vectors) `x`. |
| Y | `rt::linalg::eig` | [`eig`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eig.html) | Returns eigenvalues and eigenvectors of a real or complex matrix (or stack of matrices) `x`. (optional extension) |
| Y | `rt::linalg::eigvals` | [`eigvals`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigvals.html) | Returns the eigenvalues of a real or complex matrix (or a stack of matrices) `x`. (optional extension) |

**Notes**

- `solve` is fulfilled by `rt::linalg::solve_general` (general square `A`), which
  matches the Array API `solve`. `rt::linalg::solve_symmetric` and
  `rt::linalg::solve_triangular` additionally exploit matrix structure and can be
  faster; `solve_symmetric` is implemented by the BLAS devices only.
- `eigh` / `eig` / `eigvalsh` / `eigvals` fulfill the standard (non-generalized) form; the
  generalized eigenvalue problem is an RSTSR extension via `eigh((a, b, uplo, type))` for
  Hermitian pencils, and via `eig((a, b, side))` / `eigvals((a, b))` (`A v = λ B v`) for
  general ones (BLAS devices only).
- `eig` / `eigvals` always return complex eigenvalues (and, for `eig`, the requested
  `EigSide::Left` / `EigSide::Right` eigenvector blocks, `EigSide::Right` by default),
  in backend order (unsorted). `eigvals((a, Homogeneous))` returns the `(α, β)` pair
  stacked as `(..., 2, n)` instead of `λ = α / β`. The `EigSide` and `Homogeneous` forms
  are available on the BLAS devices only.
- `rt::linalg::lstsq` (NumPy/SciPy least squares, returning `(x, residuals, rank, s)`)
  is an RSTSR extension outside the standard; it is implemented by the BLAS
  devices only.
- The `rt::linalg::*` functions require the `linalg` cargo feature.

## Fast Fourier Transform (extensions)

| status | implementation | Python API | description |
|-|-|-|-|
|   |   | [`fft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.fft.html) | Computes the one-dimensional discrete Fourier transform. |
|   |   | [`ifft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.ifft.html) | Computes the one-dimensional inverse discrete Fourier transform. |
|   |   | [`fftn`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.fftn.html) | Computes the n-dimensional discrete Fourier transform. |
|   |   | [`ifftn`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.ifftn.html) | Computes the n-dimensional inverse discrete Fourier transform. |
|   |   | [`rfft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.rfft.html) | Computes the one-dimensional discrete Fourier transform for real-valued input. |
|   |   | [`irfft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.irfft.html) | Computes the one-dimensional inverse of `rfft` for complex-valued input. |
|   |   | [`rfftn`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.rfftn.html) | Computes the n-dimensional discrete Fourier transform for real-valued input. |
|   |   | [`irfftn`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.irfftn.html) | Computes the n-dimensional inverse of `rfftn` for complex-valued input. |
|   |   | [`hfft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.hfft.html) | Computes the one-dimensional discrete Fourier transform of a signal with Hermitian symmetry. |
|   |   | [`ihfft`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.ihfft.html) | Computes the one-dimensional inverse discrete Fourier transform of a signal with Hermitian symmetry. |
|   |   | [`fftfreq`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.fftfreq.html) | Computes the discrete Fourier transform sample frequencies. |
|   |   | [`rfftfreq`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.rfftfreq.html) | Computes the discrete Fourier transform sample frequencies (for `rfft` and `irfft`). |
|   |   | [`fftshift`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.fftshift.html) | Shifts the zero-frequency component to the center of the spectrum. |
|   |   | [`ifftshift`](https://data-apis.org/array-api/latest/extensions/generated/array_api.fft.ifftshift.html) | Inverse of `fftshift`. |
