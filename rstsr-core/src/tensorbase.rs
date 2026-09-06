//! Fundamental tensor types: [`TensorBase`] and the ownership aliases
//! specialized from it ([`Tensor`], [`TensorView`], [`TensorMut`],
//! [`TensorCow`], [`TensorArc`]), plus the fully generic alias [`TensorAny`].
//!
//! A tensor is a pair of (storage, layout): the storage holds raw data and
//! the device that owns it; the layout describes shape, stride, and offset.
//! See [`api_specification`](crate::api_specification) for the complete type
//! map, and [`Layout`] for layout semantics.

use crate::prelude_dev::*;

/// Marker trait for tensor basic types ([`TensorBase`] and its aliases).
pub trait TensorBaseAPI {}

/// The basic struct of a tensor: storage (raw data and its device) plus
/// layout (shape, stride, and offset).
///
/// Users rarely name [`TensorBase`] directly; everyday code uses the
/// ownership aliases specialized from it: [`Tensor`] (owns its data),
/// [`TensorView`] / [`TensorMut`] (borrow its data), [`TensorCow`]
/// (copy-on-write), and [`TensorArc`] (shared by atomic reference counting).
/// Functions of this crate are generic over [`TensorBase`] specializations,
/// so owned tensors and views share the same API surface.
///
/// See also [`api_specification`](crate::api_specification) for the tensor
/// structure figure and the ownership tables.
pub struct TensorBase<S, D>
where
    D: DimAPI,
{
    pub(crate) storage: S,
    pub(crate) layout: Layout<D>,
}

/// Tensor that owns its raw data (default backend [`DeviceCpu`], dynamic
/// dimensionality [`IxD`]).
///
/// This is the most common tensor kind: creation functions ([`zeros`],
/// [`arange`], ...) and computations return it. Conversions that consume or
/// change ownership ([`TensorAny::into_owned`], [`TensorAny::into_shared`],
/// [`TensorAny::into_cow`], ...) are listed in
/// [`ownership_conversion`](crate::tensor::ownership_conversion).
pub type Tensor<T, B = DeviceCpu, D = IxD> = TensorBase<Storage<DataOwned<<B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Tensor that shares its raw data by immutable reference; also available
/// under the alias [`TensorRef`].
///
/// Views are the cheapest tensor kind: slicing ([`slice`](slice())),
/// [`TensorAny::view`], and layout-only manipulations return them, and
/// creating or dropping a view never touches the underlying data. A view
/// becomes owned data by [`TensorAny::into_owned`], or copies by
/// [`TensorAny::to_owned`].
pub type TensorView<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataRef<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Tensor that shares its raw data by mutable reference; also available under
/// the alias [`TensorMut`].
///
/// Created by [`TensorAny::view_mut`], [`slice_mut`], and related functions.
/// Writing through a mutable view directly modifies the underlying data of
/// the tensor it was borrowed from.
pub type TensorViewMut<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataMut<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Tensor that either shares its raw data by reference or owns it; `Cow`
/// refers to copy-on-write.
///
/// Returned by conditional-copy conversions such as [`reshape`]/[`to_shape`]:
/// no data is copied while the requested layout can be served by a view;
/// otherwise the buffer is cloned immediately and an owned result is returned.
/// A view-backed result is cloned later only when converted to an owned tensor
/// ([`TensorAny::into_owned`]). Also created by [`TensorAny::into_cow`].
pub type TensorCow<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataCow<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Tensor whose raw data is wrapped in an atomically reference-counted
/// pointer (shared ownership).
///
/// Created by [`TensorAny::into_shared`]. Sharing is cheap (no data is moved
/// or copied), and a mutable view of shared data clones the buffer only when
/// the data is actually shared (copy-on-write).
pub type TensorArc<T, B = DeviceCpu, D = IxD> = TensorBase<Storage<DataArc<<B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Tensor that holds raw data by an enum of immutable or mutable reference;
/// an internal type used by device-level operation parameter structs.
pub type TensorReference<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataReference<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;

/// Fully generic tensor alias: [`TensorBase`] specialized by the storage
/// representation `R` (how raw data is owned), dtype `T`, device `B`, and
/// dimensionality `D`.
///
/// Function signatures are commonly written against [`TensorAny`]; each
/// ownership alias fixes `R` to one representation: [`Tensor`] uses owned
/// data, [`TensorView`] an immutable reference, [`TensorMut`] a mutable
/// reference, [`TensorCow`] copy-on-write data, and [`TensorArc`] an
/// atomically reference-counted pointer.
pub type TensorAny<R, T, B, D> = TensorBase<Storage<R, T, B>, D>;
pub use TensorView as TensorRef;
pub use TensorViewMut as TensorMut;

impl<R, D> TensorBaseAPI for TensorBase<R, D> where D: DimAPI {}

/// Basic definitions for tensor object.
impl<S, D> TensorBase<S, D>
where
    D: DimAPI,
{
    /// Initialize tensor object.
    ///
    /// # Safety
    ///
    /// This function will not check whether data meets the standard of
    /// [Storage<T, B>], or whether layout may exceed pointer bounds of data.
    pub unsafe fn new_unchecked(storage: S, layout: Layout<D>) -> Self {
        Self { storage, layout }
    }

    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    #[inline]
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }

    #[inline]
    pub fn shape(&self) -> &D {
        self.layout().shape()
    }

    #[inline]
    pub fn stride(&self) -> &D::Stride {
        self.layout().stride()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout().offset()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.layout().size()
    }

    #[inline]
    pub fn into_data(self) -> S {
        self.storage
    }

    #[inline]
    pub fn into_raw_parts(self) -> (S, Layout<D>) {
        (self.storage, self.layout)
    }

    #[inline]
    pub fn c_contig(&self) -> bool {
        self.layout().c_contig()
    }

    #[inline]
    pub fn f_contig(&self) -> bool {
        self.layout().f_contig()
    }

    #[inline]
    pub fn c_prefer(&self) -> bool {
        self.layout().c_prefer()
    }

    #[inline]
    pub fn f_prefer(&self) -> bool {
        self.layout().f_prefer()
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn new_f(storage: Storage<R, T, B>, layout: Layout<D>) -> Result<Self> {
        // check stride sanity
        layout.check_strides(true)?;

        // check pointer exceed
        let len_data = storage.len();
        let (_, idx_max) = layout.bounds_index()?;
        rstsr_pattern!(idx_max, ..=len_data, ValueOutOfRange)?;
        return Ok(Self { storage, layout });
    }

    pub fn new(storage: Storage<R, T, B>, layout: Layout<D>) -> Self {
        Self::new_f(storage, layout).rstsr_unwrap()
    }

    pub fn device(&self) -> &B {
        self.storage().device()
    }

    pub fn device_mut(&mut self) -> &mut B {
        self.storage_mut().device_mut()
    }

    pub fn data(&self) -> &R {
        self.storage().data()
    }

    pub fn data_mut(&mut self) -> &mut R {
        self.storage_mut().data_mut()
    }

    pub fn raw(&self) -> &B::Raw {
        self.storage().data().raw()
    }

    pub fn raw_mut(&mut self) -> &mut B::Raw
    where
        R: DataMutAPI<Data = B::Raw>,
    {
        self.storage_mut().data_mut().raw_mut()
    }
}

impl<T, B, D> TensorCow<'_, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn is_owned(&self) -> bool {
        self.data().is_owned()
    }

    pub fn is_ref(&self) -> bool {
        self.data().is_ref()
    }
}

/* #region TensorReference */

impl<T, B, D> TensorReference<'_, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn is_ref(&self) -> bool {
        self.data().is_ref()
    }

    pub fn is_mut(&self) -> bool {
        self.data().is_mut()
    }
}

impl<'a, T, B, D> From<TensorView<'a, T, B, D>> for TensorReference<'a, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    fn from(tensor: TensorView<'a, T, B, D>) -> Self {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = DataReference::Ref(data);
        let storage = Storage::new(data, device);
        TensorReference::new(storage, layout)
    }
}

impl<'a, T, B, D> From<TensorViewMut<'a, T, B, D>> for TensorReference<'a, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    fn from(tensor: TensorViewMut<'a, T, B, D>) -> Self {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = DataReference::Mut(data);
        let storage = Storage::new(data, device);
        TensorReference::new(storage, layout)
    }
}

impl<'a, T, B, D> From<TensorReference<'a, T, B, D>> for TensorView<'a, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    fn from(tensor: TensorReference<'a, T, B, D>) -> Self {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = match data {
            DataReference::Ref(data) => data,
            DataReference::Mut(_) => {
                rstsr_raise!(RuntimeError, "cannot convert to TensorView if data is mutable").rstsr_unwrap()
            },
        };
        let storage = Storage::new(data, device);
        TensorView::new(storage, layout)
    }
}

impl<'a, T, B, D> From<TensorReference<'a, T, B, D>> for TensorMut<'a, T, B, D>
where
    B: DeviceAPI<T>,
    D: DimAPI,
{
    fn from(tensor: TensorReference<'a, T, B, D>) -> Self {
        let (storage, layout) = tensor.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = match data {
            DataReference::Mut(data) => data,
            DataReference::Ref(_) => {
                rstsr_raise!(RuntimeError, "cannot convert to TensorMut if data is immutable").rstsr_unwrap()
            },
        };
        let storage = Storage::new(data, device);
        TensorViewMut::new(storage, layout)
    }
}

/* #endregion */

unsafe impl<R, D> Send for TensorBase<R, D>
where
    D: DimAPI,
    R: Send,
{
}

unsafe impl<R, D> Sync for TensorBase<R, D>
where
    D: DimAPI,
    R: Sync,
{
}
