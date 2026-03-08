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

## Linear Algebra (extensions)

| status | implementation | Python API | description |
|-|-|-|-|
|   |   | [`cholesky`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cholesky.html) | Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix `x`. |
|   |   | [`cross`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cross.html) | Returns the cross product of 3-element vectors. |
|   |   | [`det`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.det.html) | Returns the determinant of a square matrix (or a stack of square matrices) `x`. |
|   |   | [`diagonal`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.diagonal.html) | Returns the specified diagonals of a matrix (or a stack of matrices) `x`. |
|   |   | [`eigh`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigh.html) | Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) `x`. |
|   |   | [`eigvalsh`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigvalsh.html) | Returns the eigenvalues of a complex Hermitian or real symmetric matrix (or a stack of matrices) `x`. |
|   |   | [`inv`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.inv.html) | Returns the multiplicative inverse of a square matrix (or a stack of square matrices) `x`. |
|   |   | [`matmul`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matmul.html) | Computes the matrix product. |
|   |   | [`matrix_norm`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_norm.html) | Computes the matrix norm of a matrix (or a stack of matrices) `x`. |
|   |   | [`matrix_power`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_power.html) | Raises a square matrix (or a stack of square matrices) `x` to an integer power `n`. |
|   |   | [`matrix_rank`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_rank.html) | Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices). |
|   |   | [`matrix_transpose`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_transpose.html) | Transposes a matrix (or a stack of matrices) x. |
|   |   | [`outer`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.outer.html) | Returns the outer product of two vectors `x1` and `x2`. |
|   |   | [`pinv`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.pinv.html) | Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) `x`. |
|   |   | [`qr`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.qr.html) | Returns the QR decomposition of a full column rank matrix (or a stack of matrices). |
|   |   | [`slogdet`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.slogdet.html) | Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) `x`. |
|   |   | [`solve`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.solve.html) | Returns the solution of a square system of linear equations with a unique solution. |
|   |   | [`svd`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svd.html) | Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) `x`. |
|   |   | [`svdvals`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svdvals.html) | Returns the singular values of a matrix (or a stack of matrices) `x`. |
|   |   | [`tensordot`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.tensordot.html) | Returns a tensor contraction of x1 and x2 over specific axes. |
|   |   | [`trace`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.trace.html) | Returns the sum along the specified diagonals of a matrix (or a stack of matrices) `x`. |
|   |   | [`vecdot`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.vecdot.html) | Computes the (vector) dot product of two arrays. |
|   |   | [`vector_norm`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.vector_norm.html) | Computes the vector norm of a vector (or batch of vectors) `x`. |
|   |   | [`eig`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eig.html) | Returns eigenvalues and eigenvectors of a real or complex matrix (or stack of matrices) `x`. (optional extension) |
|   |   | [`eigvals`](https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigvals.html) | Returns the eigenvalues of a real or complex matrix (or a stack of matrices) `x`. (optional extension) |
