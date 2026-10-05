use crate::prelude_dev::*;
use num::Complex;

/// Dtype-conversion of a tensor with a compile-time target dtype.
///
/// Implemented per (source dtype, target dtype) pair of the dtype traits;
/// pairs without a cast (`DTypeCastAPI`) are simply not usable. The identity
/// pair (`Self == TOut`) is view-backed: [`astype`] borrows the input and
/// returns a zero-copy view, and [`into_astype`] moves the storage.
///
/// This trait is used as the dispatch mechanism of [`astype`]; users should
/// need to reference it only to add conversion support for custom dtypes.
pub trait DTypeAstypeAPI<TOut>
where
    Self: Sized,
{
    /// Convert the element dtype of a tensor, reusing the buffer when the
    /// source and target dtypes are the same.
    #[allow(clippy::type_complexity)]
    fn astype_cow<'a, R, B, D>(tensor: &'a TensorAny<R, Self, B, D>) -> Result<TensorCow<'a, TOut, B, D>>
    where
        R: DataAPI<Data = <B as DeviceRawAPI<Self>>::Raw> + DataIntoCowAPI<'a>,
        D: DimAPI,
        B: DeviceAPI<Self>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, Self>;

    /// Consuming counterpart of [`DTypeAstypeAPI::astype_cow`]; when the
    /// dtypes match, the storage is moved instead of copied.
    #[allow(clippy::type_complexity)]
    fn into_astype<R, B, D>(tensor: TensorAny<R, Self, B, D>) -> Result<Tensor<TOut, B, D>>
    where
        R: DataAPI<Data = <B as DeviceRawAPI<Self>>::Raw> + DataCloneAPI,
        R::Data: Clone,
        D: DimAPI,
        Self: Clone,
        B: DeviceAPI<Self>
            + DeviceRawAPI<MaybeUninit<Self>>
            + DeviceCreationAnyAPI<Self>
            + OpAssignAPI<Self, D>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, Self>;
}

/// Between different dtypes: allocate a fresh buffer and cast on the fly.
macro_rules! impl_astype_cast {
    ($T:ty, $TOut:ty) => {
        impl DTypeAstypeAPI<$TOut> for $T {
            fn astype_cow<'a, R, B, D>(tensor: &'a TensorAny<R, $T, B, D>) -> Result<TensorCow<'a, $TOut, B, D>>
            where
                R: DataAPI<Data = <B as DeviceRawAPI<$T>>::Raw> + DataIntoCowAPI<'a>,
                D: DimAPI,
                B: DeviceAPI<$T>
                    + DeviceRawAPI<MaybeUninit<$TOut>>
                    + DeviceAPI<$TOut>
                    + DeviceCreationAnyAPI<$TOut>
                    + OpAssignArbitaryAPI<$TOut, D, D, $T>,
            {
                let la = tensor.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let device = tensor.device();
                let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
                device.assign_arbitary_uninit(storage_c.raw_mut(), &lc, tensor.raw(), la)?;
                // SAFETY: `assign_arbitary_uninit` wrote every element of the
                // fresh contiguous storage above.
                let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
                Ok(Tensor::new_f(storage_c, lc)?.into_cow())
            }

            fn into_astype<R, B, D>(tensor: TensorAny<R, $T, B, D>) -> Result<Tensor<$TOut, B, D>>
            where
                R: DataAPI<Data = <B as DeviceRawAPI<$T>>::Raw> + DataCloneAPI,
                R::Data: Clone,
                D: DimAPI,
                $T: Clone,
                B: DeviceAPI<$T>
                    + DeviceRawAPI<MaybeUninit<$T>>
                    + DeviceCreationAnyAPI<$T>
                    + OpAssignAPI<$T, D>
                    + DeviceRawAPI<MaybeUninit<$TOut>>
                    + DeviceAPI<$TOut>
                    + DeviceCreationAnyAPI<$TOut>
                    + OpAssignArbitaryAPI<$TOut, D, D, $T>,
            {
                let la = tensor.layout();
                let lc = layout_for_array_copy(la, TensorIterOrder::default())?;
                let device = tensor.device();
                let mut storage_c = device.uninit_impl(lc.bounds_index()?.1)?;
                device.assign_arbitary_uninit(storage_c.raw_mut(), &lc, tensor.raw(), la)?;
                // SAFETY: `assign_arbitary_uninit` wrote every element of the
                // fresh contiguous storage above.
                let storage_c = unsafe { B::assume_init_impl(storage_c) }?;
                Tensor::new_f(storage_c, lc)
            }
        }
    };
}

/// Emit both directions of a dtype pair.
macro_rules! impl_astype_pair {
    ($T:ty; $($U:ty),* $(,)?) => {$(
        impl_astype_cast!($T, $U);
        impl_astype_cast!($U, $T);
    )*};
}

/// Emit all ordered, non-identical pairs of a dtype list.
macro_rules! impl_astype_all_pairs {
    ([]) => {};
    ([$head:ty $(, $tail:ty)* $(,)?]) => {
        impl_astype_pair!($head; $($tail),*);
        impl_astype_all_pairs!([$($tail),*]);
    };
}

/// Same dtype: return a view of the input (no copy).
impl<T> DTypeAstypeAPI<T> for T
where
    T: Clone,
{
    fn astype_cow<'a, R, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, T, B, D>>
    where
        R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
        D: DimAPI,
        B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D, T>,
    {
        Ok(tensor.view().into_cow())
    }

    fn into_astype<R, B, D>(tensor: TensorAny<R, T, B, D>) -> Result<Tensor<T, B, D>>
    where
        R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataCloneAPI,
        R::Data: Clone,
        D: DimAPI,
        T: Clone,
        B: DeviceAPI<T>
            + DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + OpAssignAPI<T, D>
            + DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + OpAssignArbitaryAPI<T, D, D, T>,
    {
        Ok(tensor.into_owned())
    }
}

impl_astype_all_pairs!([
    u8, u16, u32, u64, i8, i16, i32, i64, usize, isize, f32, f64, bool, Complex<f32>, Complex<f64>,
]);

/// Cast a tensor to another dtype (borrowing).
///
/// The input is not copied when `TOut` is the same as the input dtype: the
/// result is then a zero-copy view ([`TensorCow`]). Otherwise the elements are
/// converted into a fresh buffer with the same shape, contiguous in the
/// device's default order.
///
/// `TOut` is a type parameter; spell it with the associated-method form
/// `a.astype::<f64>()`, or annotate the free function's result:
/// `let b: TensorCow<f64, _, _> = rt::astype(&a);`.
///
/// # Parameters
///
/// - `tensor`: the input tensor.
///
/// # Returns
///
/// - [`TensorCow<'_, TOut, B, D>`][`TensorCow`]: the converted tensor (or a view of the input).
///
/// `TOut` is a type parameter: use the associated-method form `a.astype::<f64>()` (single
/// turbofish), or annotate the result of the free function:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
///
/// // method form (recommended)
/// let b = a.astype::<f64>();
/// println!("{b}");
/// // [[ 1 2]
/// //  [ 3 4]]
///
/// // free function with a type annotation
/// let c: TensorCow<f64, _, _> = rt::astype(&a);
/// # assert_eq!(format!("{b}"), "[[ 1 2]\n [ 3 4]]");
/// # assert_eq!(b.raw(), &vec![1.0f64, 2.0, 3.0, 4.0]);
/// # assert_eq!(c.raw(), &vec![1.0f64, 2.0, 3.0, 4.0]);
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `astype(x, dtype, /, *, copy=True)` ([`astype`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.astype.html))
/// - NumPy: `ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)` ([`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html))
/// - RSTSR: `rt::astype::<TOut>(&x)`; the dtype is a type parameter rather than a runtime value,
///   and copying is conditional (view when the dtype is unchanged), so no `copy` argument is
///   needed.
///
/// Note that `astype` does not follow the `to_`/`into_` naming convention of this crate: a dtype
/// change always copies, while `to_*` functions return views. Prefer this explicit exception over
/// ad-hoc conversions; `to_*` remains reserved for layout-only operations.
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)
///
/// ## Variants of this function
///
/// - [`astype`] / [`astype_f`]: taking reference and returning [`TensorCow`].
/// - [`into_astype`] / [`into_astype_f`]: consuming version.
pub fn astype_f<'a, TOut, T, R, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, TOut, B, D>>
where
    T: DTypeAstypeAPI<TOut>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceAPI<TOut>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    T::astype_cow(tensor)
}

/// Cast a tensor to another dtype (borrowing).
///
/// See also [`astype`].
pub fn astype<'a, TOut, T, R, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> TensorCow<'a, TOut, B, D>
where
    T: DTypeAstypeAPI<TOut>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceAPI<TOut>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    astype_f(tensor).rstsr_unwrap()
}

/// Cast a tensor to another dtype (consuming).
///
/// When `TOut` is the same as the input dtype, the storage is moved rather
/// than copied.
///
/// See also [`astype`].
pub fn into_astype_f<TOut, T, R, B, D>(tensor: TensorAny<R, T, B, D>) -> Result<Tensor<TOut, B, D>>
where
    T: DTypeAstypeAPI<TOut>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataCloneAPI,
    R::Data: Clone,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceAPI<TOut>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    T::into_astype(tensor)
}

/// Cast a tensor to another dtype (consuming).
///
/// See also [`astype`].
pub fn into_astype<TOut, T, R, B, D>(tensor: TensorAny<R, T, B, D>) -> Tensor<TOut, B, D>
where
    T: DTypeAstypeAPI<TOut>,
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataCloneAPI,
    R::Data: Clone,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T>
        + DeviceRawAPI<MaybeUninit<T>>
        + DeviceCreationAnyAPI<T>
        + OpAssignAPI<T, D>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceAPI<TOut>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    into_astype_f(tensor).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Cast the tensor to another dtype (borrowing).
    ///
    /// See also [`astype`].
    pub fn astype<'a, TOut>(&'a self) -> TensorCow<'a, TOut, B, D>
    where
        T: DTypeAstypeAPI<TOut>,
        R: DataIntoCowAPI<'a>,
        B: DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        astype_f(self).rstsr_unwrap()
    }

    /// Cast the tensor to another dtype (borrowing).
    ///
    /// See also [`astype`].
    pub fn astype_f<'a, TOut>(&'a self) -> Result<TensorCow<'a, TOut, B, D>>
    where
        T: DTypeAstypeAPI<TOut>,
        R: DataIntoCowAPI<'a>,
        B: DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        astype_f(self)
    }

    /// Cast the tensor to another dtype (consuming).
    ///
    /// See also [`astype`].
    pub fn into_astype<TOut>(self) -> Tensor<TOut, B, D>
    where
        T: DTypeAstypeAPI<TOut> + Clone,
        R: DataCloneAPI,
        R::Data: Clone,
        B: DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + OpAssignAPI<T, D>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        into_astype_f(self).rstsr_unwrap()
    }

    /// Cast the tensor to another dtype (consuming).
    ///
    /// See also [`astype`].
    pub fn into_astype_f<TOut>(self) -> Result<Tensor<TOut, B, D>>
    where
        T: DTypeAstypeAPI<TOut> + Clone,
        R: DataCloneAPI,
        R::Data: Clone,
        B: DeviceRawAPI<MaybeUninit<T>>
            + DeviceCreationAnyAPI<T>
            + OpAssignAPI<T, D>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceAPI<TOut>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        into_astype_f(self)
    }
}
