use crate::prelude_dev::*;

/// Convert layout to the other dimension.
///
/// See also [`to_dim`].
pub fn into_dim_f<S, D, D2>(tensor: TensorBase<S, D>) -> Result<TensorBase<S, D2>>
where
    D: DimAPI + DimIntoAPI<D2>,
    D2: DimAPI,
{
    let (storage, layout) = tensor.into_raw_parts();
    let layout = layout.into_dim::<D2>()?;
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Convert layout to the other dimension.
///
/// This function converts the dimensionality of a tensor's layout to a target dimension type.
/// No data is copied; only the layout representation is modified.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// - generic parameter `D2`: [`DimAPI`]
///
///   - Target dimension type.
///   - [`IxD`] or [`IxDyn`] (equilvalent to [`Vec<usize>`]) can be used for dynamic dimension.
///   - [`Ix<N>`] (given `const N: usize`, equilvalent to [`[usize; N]`](https://doc.rust-lang.org/core/primitive.array.html))
///     can be used for static dimension.
///   - Dynamic dimensionality is usually more preferred for general use cases, while static
///     dimensionality is more suitable for performance-critical code, especially frequent indexing
///     and non-contiguous memory access.
///
/// # Returns
///
/// - [`TensorView<'_, T, B, D2>`](TensorView)
///
///   - A view of the input tensor with the layout converted to the target dimension type.
///   - The underlying data is not copied; only the layout of the view is modified.
///   - If you want to convert the tensor itself (taking the ownership instead of returning view),
///     use [`into_dim`](TensorAny::into_dim) instead.
///
/// # Examples
///
/// In most cases, RSTSR will generate dynamic dimension ([`IxD`]) in most cases, including indexing
/// [`slice`](slice()), reshaping [`reshape`], creation [`asarray`].
///
/// You can debug print tensor or it's layout to verify the dimensionality, or call `const_ndim` to
/// check whether the dimension is static or dynamic.
///
/// ```rust
/// use rstsr::prelude::*;
/// let a = rt::arange(6).into_shape([2, 3]); // shape: (2, 3), IxD
/// println!("a: {:?}", a);
/// // `None` here indicates dynamic dimension
/// assert_eq!(a.shape().const_ndim(), None);
/// ```
///
/// You can convert the dimension to static dimension by associate method:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let a = rt::arange(6).into_shape([2, 3]); // shape: (2, 3), IxD
/// let b = a.to_dim::<Ix2>(); // shape: (2, 3), Ix2
/// assert_eq!(b.shape().const_ndim(), Some(2));
/// ```
///
/// You can use `.to_dim::<IxD>()` or `.to_dyn()` to convert back to dynamic dimension:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let a = rt::arange(6).into_shape([2, 3]); // shape: (2, 3), IxD
/// # let b = a.to_dim::<Ix2>(); // shape: (2, 3), Ix2
/// let c = b.to_dyn(); // shape: (2, 3), IxD
/// assert_eq!(c.shape().const_ndim(), None);
/// ```
///
/// # Panics
///
/// - The shape of the tensor is not compatible with the target dimension type.
///
/// ```rust,should_panic
/// use rstsr::prelude::*;
/// let a = rt::arange(6).into_shape([2, 3]); // shape: (2, 3), IxD
/// let b = a.to_dim::<Ix3>(); // shape: (2, 3), Ix3, panics
/// ```
///
/// # See also
///
/// ## Similar functions from other crates/libraries
///
/// - ndarray: [`into_dimensionality`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.into_dimensionality)
/// - ndarray: [`into_dyn`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.into_dyn)
///
/// ## Variants of this function
///
/// - [`to_dim`] / [`to_dim_f`]: Returning a view.
/// - [`into_dim`] / [`into_dim_f`]: Consuming version.
/// - [`to_dyn`]: Convert to dynamic dimension, returning a view (infallable by definition).
/// - [`into_dyn`]: Convert to dynamic dimension, consuming version (infallible by definition).
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::to_dim`] / [`TensorAny::to_dim_f`]
///   - [`TensorAny::into_dim`] / [`TensorAny::into_dim_f`]
///   - [`TensorAny::to_dyn`]
///   - [`TensorAny::into_dyn`]
pub fn to_dim<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, D2>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view()).rstsr_unwrap()
}

/// Convert layout to the other dimension.
///
/// See also [`to_dim`].
pub fn to_dim_f<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view())
}

/// Convert layout to the other dimension.
///
/// See also [`to_dim`].
pub fn into_dim<S, D, D2>(tensor: TensorBase<S, D>) -> TensorBase<S, D2>
where
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
{
    into_dim_f(tensor).rstsr_unwrap()
}

/// Convert layout to the dynamic dimension [`IxD`].
///
/// See also [`to_dim`].
pub fn to_dyn<R, T, B, D>(tensor: &TensorAny<R, T, B, D>) -> TensorView<'_, T, B, IxD>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_dim_f(tensor.view()).rstsr_unwrap()
}

/// Convert layout to the dynamic dimension [`IxD`].
///
/// See also [`to_dim`].
pub fn into_dyn<S, D>(tensor: TensorBase<S, D>) -> TensorBase<S, IxD>
where
    D: DimAPI,
{
    into_dim_f(tensor).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    /// Convert layout to the other dimension.
    ///
    /// See also [`to_dim`].
    pub fn to_dim<D2>(&self) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        to_dim(self)
    }

    /// Convert layout to the other dimension.
    ///
    /// See also [`to_dim`].
    pub fn to_dim_f<D2>(&self) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        to_dim_f(self)
    }

    /// Convert layout to the other dimension.
    ///
    /// See also [`to_dim`].
    pub fn into_dim<D2>(self) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim(self)
    }

    /// Convert layout to the other dimension.
    ///
    /// See also [`to_dim`].
    pub fn into_dim_f<D2>(self) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim_f(self)
    }

    /// Convert layout to the dynamic dimension [`IxD`].
    ///
    /// See also [`to_dim`].
    pub fn to_dyn(&self) -> TensorView<'_, T, B, IxD> {
        to_dyn(self)
    }

    /// Convert layout to the dynamic dimension [`IxD`].
    ///
    /// See also [`to_dim`].
    pub fn into_dyn(self) -> TensorAny<R, T, B, IxD> {
        into_dyn(self)
    }
}
