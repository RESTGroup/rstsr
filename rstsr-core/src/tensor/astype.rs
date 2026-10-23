//! Data type conversion: [`astype`] / [`astype_f`] and their associated
//! methods.

use crate::prelude_dev::*;
use core::any::TypeId;

/* #region astype */

/// Casts a tensor to another data type.
///
/// See also [`astype`].
pub fn astype_f<'a, TOut, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> Result<TensorCow<'a, TOut, B, D>>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: 'static,
    TOut: 'static,
    B: DeviceAPI<T>
        + DeviceAPI<TOut>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    // same dtype: return a view of the input buffer
    if TypeId::of::<T>() == TypeId::of::<TOut>() {
        // SAFETY: `TypeId` equality proves `T == TOut`, so the device raw
        // container types are identical; only the dtype marker changes.
        let raw =
            unsafe { &*(tensor.raw() as *const <B as DeviceRawAPI<T>>::Raw as *const <B as DeviceRawAPI<TOut>>::Raw) };
        let storage = Storage::new(DataCow::Ref(DataRef::TrueRef(raw)), tensor.device().clone());
        return TensorBase::new_f(storage, tensor.layout().clone());
    }
    // different dtype: allocate a dense buffer in the input's memory order and
    // cast element-wise by the device assign kernel
    let la = tensor.layout();
    let lc = layout_for_array_copy(la, TensorIterOrder::K)?;
    let device = tensor.device();
    let mut storage = device.uninit_impl(lc.bounds_index()?.1)?;
    device.assign_arbitary_uninit(storage.raw_mut(), &lc, tensor.raw(), la)?;
    // SAFETY: `assign_arbitary_uninit` wrote every element of `lc`, which covers
    // the fresh storage sized to its bounds.
    let storage = unsafe { B::assume_init_impl(storage)? };
    let (data, device) = storage.into_raw_parts();
    TensorBase::new_f(Storage::new(DataCow::Owned(data), device), lc)
}

/// Casts a tensor to another data type.
///
/// When the target dtype equals the input dtype, no data is copied: the result
/// is a view of the input keeping its layout (shape, strides and offset)
/// unchanged. Otherwise every element is converted to `TOut` and stored in a
/// newly allocated dense tensor whose axes keep the input's memory order (the
/// `order='K'` convention).
///
/// The element conversion is provided by the device backend through
/// `DTypeCastAPI`; every real/bool/complex pair is implemented except
/// complex-to-real, which the Array API forbids outright (convert through
/// `real` or `imag` first to state which component is meant). The cast follows
/// Rust's `as` semantics: float-to-int saturates, integer-to-integer
/// conversions wrap, and out-of-range values are implementation-defined by the
/// Array API.
///
/// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
///
/// # Parameters
///
/// - `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny)
///
///   - The input tensor.
///
/// # Returns
///
/// - [`TensorCow<'_, TOut, B, D>`](TensorCow)
///
///   - A view when `TOut` is the same dtype as the input; otherwise a newly allocated tensor with
///     converted elements.
///
/// # Examples
///
/// Cast an integer tensor to floating point; a same-dtype call returns a view
/// without allocating:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::arange((6, &device)).into_shape([2, 3]);
/// let b = a.astype::<f64>();
/// println!("{b}");
/// // [[ 0 1 2]
/// //  [ 3 4 5]]
/// # assert!(b.is_owned());
///
/// // same dtype: a view, no allocation
/// let c = a.astype::<i32>();
/// # assert!(!c.is_owned());
/// # let expected = rt::tensor_from_nested!([[0f64, 1., 2.], [3., 4., 5.]], &device);
/// # assert!(rt::allclose(&b, &expected, None));
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `astype(x, dtype, /, *, copy=True, device=None)` ([`astype`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.astype.html))
/// - NumPy: `ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)` ([`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html))
/// - RSTSR: `tensor.astype::<TOut>()`
///
/// The associated method is the preferred spelling; the free-function form
/// `rt::astype` takes the same generic parameters as this function, so the
/// remaining ones have to be inferred or supplied explicitly.
///
/// This function takes no `copy` or `device` argument: a same-dtype conversion
/// is always a view (NumPy's `copy=False`), while a dtype-changing conversion
/// always allocates a dense copy in the input's memory order (NumPy's
/// `order='K'`). If an explicit copy is required, call
/// [`TensorAny::into_owned`] on the result. NumPy's `order`, `casting` and
/// `subok` arguments are not supported.
///
/// # Panics
///
/// - Panics if the output tensor cannot be allocated, or if the internal copy fails (for example on
///   an overflowing element count).
///
/// For a fallible version, use [`astype_f`].
///
/// # See also
///
/// ## Similar function from other crates/libraries
///
/// - NumPy: [`numpy.ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)
/// - Array-API: [`astype`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.astype.html)
///
/// ## Related functions in RSTSR
///
/// - [`TensorAny::mapv`]: element-wise mapping with a closure; use it for conversions outside the
///   cast matrix (for example complex-to-real via `real`).
/// - [`to_contig`]: conditional copy without a dtype change.
/// - [`reshape`]: another conversion returning [`TensorCow`], a view when the layout allows it.
///
/// ## Variants of this function
///
/// - [`astype_f`]: fallible version.
/// - Associated methods on [`TensorAny`]:
///
///   - [`TensorAny::astype`] / [`TensorAny::astype_f`]
///
/// There is deliberately no `to_astype` / `into_astype` form: the
/// copy-on-write return covers both the borrowing and the consuming case.
pub fn astype<'a, TOut, R, T, B, D>(tensor: &'a TensorAny<R, T, B, D>) -> TensorCow<'a, TOut, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    T: 'static,
    TOut: 'static,
    B: DeviceAPI<T>
        + DeviceAPI<TOut>
        + DeviceRawAPI<MaybeUninit<TOut>>
        + DeviceCreationAnyAPI<TOut>
        + OpAssignArbitaryAPI<TOut, D, D, T>,
{
    astype_f(tensor).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Casts the tensor to another data type.
    ///
    /// See also [`astype`].
    pub fn astype_f<'a, TOut>(&'a self) -> Result<TensorCow<'a, TOut, B, D>>
    where
        T: 'static,
        TOut: 'static,
        B: DeviceAPI<TOut>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        astype_f(self)
    }

    /// Casts the tensor to another data type.
    ///
    /// See also [`astype`].
    pub fn astype<'a, TOut>(&'a self) -> TensorCow<'a, TOut, B, D>
    where
        T: 'static,
        TOut: 'static,
        B: DeviceAPI<TOut>
            + DeviceRawAPI<MaybeUninit<TOut>>
            + DeviceCreationAnyAPI<TOut>
            + OpAssignArbitaryAPI<TOut, D, D, T>,
    {
        astype_f(self).rstsr_unwrap()
    }
}

/* #endregion */
