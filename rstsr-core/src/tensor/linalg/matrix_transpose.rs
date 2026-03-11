use crate::prelude_dev::*;

/* #region matrix_transpose */

/// Transposes a matrix (or a stack of matrices).
///
/// See also [`matrix_transpose`].
pub fn into_matrix_transpose_f<S, D>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
{
    into_swapaxes_f(tensor, -1, -2)
}

/// Transposes a matrix (or a stack of matrices).
///
/// Returns an array with the last two axes interchanged. This is equivalent
/// to `swapaxes(-1, -2)`, but is provided as a convenience function for
/// transposing matrices in multi-dimensional arrays.
///
/// For a 2-D array, this is equivalent to the standard matrix transpose.
/// For higher-dimensional arrays, this transposes each matrix in a stack of
/// matrices, leaving other axes unchanged.
///
/// # Examples
///
/// For a 2-D array, this is equivalent to the standard matrix transpose:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// let result = x.matrix_transpose();
/// println!("{result}");
/// // [[ 1 3]
/// //  [ 2 4]]
/// ```
///
/// For a 3-D array (a stack of matrices), each matrix is transposed independently:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let x = rt::tensor_from_nested!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
/// let result = x.matrix_transpose();
/// println!("{result}");
/// // [[[ 1 3]
/// //   [ 2 4]]
/// //  [[ 5 7]
/// //   [ 6 8]]]
/// ```
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor to be transposed.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D>`](TensorView)
///
///   - A view of the input tensor with the last two axes interchanged.
///   - No data is copied; only the shape and strides are modified.
///
/// # Notes of API accordance
///
/// - Array-API: `matrix_transpose(x)` ([`matrix_transpose`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matrix_transpose.html))
/// - NumPy: `numpy.linalg.matrix_transpose(x)` ([`numpy.matrix_transpose`](https://numpy.org/doc/stable/reference/generated/numpy.matrix_transpose.html))
/// - RSTSR: `tensor.matrix_transpose()` or `rt::matrix_transpose(&tensor)`
///
/// Note that this is different from `T` (NumPy) / `t()` (RSTSR), which reverses
/// all axes for n-dimensional arrays. This function only swaps the last two axes,
/// which corresponds to `mT` in NumPy.
///
/// # See also
///
/// ## Related functions in RSTSR
///
/// - [`transpose`] - General axis permutation
/// - [`swapaxes`] - Swap two specific axes
/// - [`reverse_axes`] - Reverse all axes order
///
/// ## Variants of this function
///
/// - [`matrix_transpose`] / [`matrix_transpose_f`]: Returning a view.
/// - [`into_matrix_transpose`] / [`into_matrix_transpose_f`]: Consuming version.
///
/// - Associated methods on `TensorAny`:
///
///   - [`TensorAny::matrix_transpose`] / [`TensorAny::matrix_transpose_f`]
///   - [`TensorAny::into_matrix_transpose`] / [`TensorAny::into_matrix_transpose_f`]
pub fn matrix_transpose<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), -1, -2).rstsr_unwrap()
}

/// Transposes a matrix (or a stack of matrices).
///
/// See also [`matrix_transpose`].
pub fn matrix_transpose_f<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_swapaxes_f(tensor.view(), -1, -2)
}

/// Transposes a matrix (or a stack of matrices).
///
/// See also [`matrix_transpose`].
pub fn into_matrix_transpose<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, D>
where
    D: DimAPI,
{
    into_swapaxes_f(tensor, -1, -2).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Transposes a matrix (or a stack of matrices).
    ///
    /// See also [`matrix_transpose`].
    pub fn matrix_transpose(&self) -> TensorView<'_, T, B, D> {
        matrix_transpose(self)
    }

    /// Transposes a matrix (or a stack of matrices).
    ///
    /// See also [`matrix_transpose`].
    pub fn matrix_transpose_f(&self) -> Result<TensorView<'_, T, B, D>> {
        matrix_transpose_f(self)
    }

    /// Transposes a matrix (or a stack of matrices).
    ///
    /// See also [`matrix_transpose`].
    pub fn into_matrix_transpose(self) -> TensorAny<R, T, B, D> {
        into_matrix_transpose(self)
    }

    /// Transposes a matrix (or a stack of matrices).
    ///
    /// See also [`matrix_transpose`].
    pub fn into_matrix_transpose_f(self) -> Result<TensorAny<R, T, B, D>> {
        into_matrix_transpose_f(self)
    }
}

/* #endregion */
