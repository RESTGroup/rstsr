//! Ownership conversions between tensor kinds: views ([`TensorAny::view`],
//! [`TensorAny::view_mut`]), owned ([`TensorAny::into_owned`]), shared
//! ([`TensorAny::into_shared`]), copy-on-write ([`TensorAny::into_cow`]), and
//! scalar/buffer extraction ([`TensorAny::to_scalar`], [`TensorAny::to_vec`]).
//!
//! All conversions keep the layout unchanged; only the data ownership (or the
//! gathered elements) differs.

use crate::prelude_dev::*;

/* #region basic conversion */

/// Methods for tensor ownership conversion.
impl<R, T, B, D> TensorAny<R, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    /// Get an immutable view of the tensor.
    ///
    /// The view shares the underlying data: reading through it always reflects
    /// the original tensor, and no data is copied. This is the cheapest way to
    /// pass a tensor to functions that only read it.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`TensorView<'_, T, B, D>`][`TensorView`]: a view sharing the data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let view = a.view();
    /// println!("{view}");
    /// // [[ 0 1 2]
    /// //  [ 3 4 5]]
    /// # assert_eq!(format!("{view}"), "[[ 0 1 2]\n [ 3 4 5]]");
    /// ```
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::view_mut`]: mutable view.
    /// - [`TensorAny::to_owned`]: independent copy of the visible elements.
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorViewAPI::view`]: trait form accepting `&Tensor` and views.
    pub fn view(&self) -> TensorView<'_, T, B, D> {
        let layout = self.layout().clone();
        let data = self.data().as_ref();
        let storage = Storage::new(data, self.device().clone());
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Get a mutable view of the tensor.
    ///
    /// Writes through the view directly modify the original tensor; no data is
    /// copied.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`TensorMut<'_, T, B, D>`][`TensorMut`]: a mutable view sharing the data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let mut a: Tensor<i32, _> = rt::arange((6, &device)).into_shape([2, 3]);
    /// let mut v = a.view_mut();
    /// v += 10;
    /// drop(v);
    /// println!("{a}");
    /// // [[ 10 11 12]
    /// //  [ 13 14 15]]
    /// # assert_eq!(format!("{a}"), "[[ 10 11 12]\n [ 13 14 15]]");
    /// ```
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::view`]: immutable view.
    /// - [`slice_mut`](crate::tensor::indexing::slice_mut()): mutable view of a subsection.
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorViewMutAPI::view_mut`]: trait form accepting `&mut Tensor`.
    pub fn view_mut(&mut self) -> TensorMut<'_, T, B, D>
    where
        R: DataMutAPI,
    {
        let device = self.device().clone();
        let layout = self.layout().clone();
        let data = self.data_mut().as_mut();
        let storage = Storage::new(data, device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Convert the tensor into copy-on-write.
    ///
    /// No data is copied: an owned tensor keeps its buffer, and a view becomes
    /// a view-backed [`TensorCow`]. The data is cloned only later, if a
    /// mutation or [`TensorAny::into_owned`] requires it.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`TensorCow<'a, T, B, D>`][`TensorCow`]: copy-on-write tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device));
    /// let cow = a.into_cow();
    /// println!("{cow}");
    /// // [ 0 1 2 3 4 5]
    /// # assert_eq!(format!("{cow}"), "[ 0 1 2 3 4 5]");
    /// ```
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`reshape`]/[`to_shape`]: conditional-copy conversions returning [`TensorCow`].
    pub fn into_cow<'a>(self) -> TensorCow<'a, T, B, D>
    where
        R: DataIntoCowAPI<'a>,
    {
        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let storage = Storage::new(data.into_cow(), device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Convert the tensor into an owned tensor, keeping the layout unchanged.
    ///
    /// The whole underlying buffer is moved if possible, or fully cloned; the
    /// visible layout is never re-gathered.
    ///
    /// # See also
    ///
    /// [`TensorAny::into_owned`] additionally handles non-compact layouts by
    /// gathering the visible elements. Prefer this function when the memory
    /// bulk is large but the visible tensor is small.
    pub fn into_owned_keep_layout(self) -> Tensor<T, B, D>
    where
        R::Data: Clone,
        R: DataCloneAPI,
    {
        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let storage = Storage::new(data.into_owned(), device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Convert the tensor into a shared ([`TensorArc`]) tensor, keeping the
    /// layout unchanged.
    ///
    /// The whole underlying buffer is moved if possible, or fully cloned; the
    /// visible layout is never re-gathered.
    ///
    /// # See also
    ///
    /// [`TensorAny::into_shared`] additionally handles non-compact layouts by
    /// gathering the visible elements. Prefer this function when the memory
    /// bulk is large but the visible tensor is small.
    pub fn into_shared_keep_layout(self) -> TensorArc<T, B, D>
    where
        R::Data: Clone,
        R: DataCloneAPI,
    {
        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let storage = Storage::new(data.into_shared(), device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    R::Data: Clone,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    /// Convert the tensor into an owned tensor.
    ///
    /// If the layout covers the whole underlying buffer (compact), the buffer is
    /// moved; otherwise the visible elements are gathered (in
    /// [`TensorIterOrder::K`] arrangement) into a fresh owned tensor. In both
    /// cases the returned tensor owns its data and the logical content is
    /// unchanged.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`Tensor<T, B, D>`][`Tensor`]: owned tensor with the same logical content.
    ///
    /// # Examples
    ///
    /// A sliced (non-compact) view is gathered into a fresh owned tensor:
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((24, &device)).into_shape([2, 3, 4]);
    /// let v = a.into_slice((.., .., 0..2));
    /// let o = v.into_owned();
    /// println!("{o}");
    /// // [[[ 0 1]
    /// //   [ 4 5]
    /// //   [ 8 9]]
    /// //
    /// //  [[ 12 13]
    /// //   [ 16 17]
    /// //   [ 20 21]]]
    /// # assert_eq!(format!("{o}"), "[[[ 0 1]\n  [ 4 5]\n  [ 8 9]]\n\n [[ 12 13]\n  [ 16 17]\n  [ 20 21]]]");
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the layout is invalid (out-of-bound bounds).
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::to_owned`]: same result from a borrowed tensor.
    /// - [`TensorAny::into_owned_keep_layout`]: move or clone the whole buffer without gathering.
    /// - [`TensorAny::into_shared`]: shared ownership instead of owned.
    ///
    /// ## Variants of this function
    ///
    /// - [`TensorIntoOwnedAPI::into_owned`]: trait form.
    pub fn into_owned(self) -> Tensor<T, B, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().rstsr_unwrap();
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == self.layout().size() {
            return self.into_owned_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K));
        }
    }

    /// Convert the tensor into a shared ([`TensorArc`]) tensor.
    ///
    /// The buffer is moved when the layout covers it entirely; otherwise the
    /// visible elements are gathered into a fresh buffer first (see
    /// [`TensorAny::into_owned`] for the same move-or-gather semantics).
    /// Sharing itself is cheap, and the data is cloned only when a mutable view
    /// of shared data is requested.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`TensorArc<T, B, D>`][`TensorArc`]: shared-ownership tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device));
    /// let shared = a.into_shared();
    /// let v1 = shared.view();
    /// println!("{v1}");
    /// // [ 0 1 2 3 4 5]
    /// # assert_eq!(format!("{v1}"), "[ 0 1 2 3 4 5]");
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the layout is invalid (out-of-bound bounds).
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::into_owned`]: owned instead of shared.
    /// - [`TensorAny::into_shared_keep_layout`]: move or clone the whole buffer without gathering.
    pub fn into_shared(self) -> TensorArc<T, B, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().rstsr_unwrap();
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == self.layout().size() {
            return self.into_shared_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K)).into_shared();
        }
    }

    /// Clone the visible elements into a new owned tensor.
    ///
    /// The original tensor is only borrowed; the result always owns freshly
    /// gathered data ([`TensorIterOrder::K`] arrangement).
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - [`Tensor<T, B, D>`][`Tensor`]: independent owned copy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device)).into_shape([2, 3]);
    /// let b = a.to_owned();
    /// println!("{b}");
    /// // [[ 0 1 2]
    /// //  [ 3 4 5]]
    /// # assert_eq!(format!("{b}"), "[[ 0 1 2]\n [ 3 4 5]]");
    /// ```
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::into_owned`]: consuming form (may move instead of copy).
    /// - [`Clone`]: `Tensor` implements `Clone` by this operation.
    pub fn to_owned(&self) -> Tensor<T, B, D> {
        self.view().into_owned()
    }
}

impl<T, B, D> Clone for Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    fn clone(&self) -> Self {
        self.to_owned()
    }
}

impl<T, B, D> Clone for TensorCow<'_, T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    fn clone(&self) -> Self {
        let tsr_owned = self.to_owned();
        let (storage, layout) = tsr_owned.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let data = data.into_cow();
        let storage = Storage::new(data, device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw> + DataForceMutAPI<B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// # Safety
    ///
    /// This function is highly unsafe, as it entirely bypasses Rust's lifetime
    /// and borrowing rules.
    pub unsafe fn force_mut(&self) -> TensorMut<'_, T, B, D> {
        let layout = self.layout().clone();
        let data = self.data().force_mut();
        let storage = Storage::new(data, self.device().clone());
        TensorBase::new_unchecked(storage, layout)
    }
}

/* #endregion */

/* #region to_raw */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn to_raw_f(&self) -> Result<<B as DeviceRawAPI<T>>::Raw> {
        rstsr_assert_eq!(self.ndim(), 1, InvalidLayout, "to_vec currently only support 1-D tensor")?;
        let device = self.device();
        let layout = self.layout().to_dim::<Ix1>()?;
        let size = layout.size();
        let mut new_storage = device.uninit_impl(size)?;
        device.assign_uninit(new_storage.raw_mut(), &[size].c(), self.raw(), &layout)?;
        let storage = unsafe { B::assume_init_impl(new_storage) }?;
        let (data, _) = storage.into_raw_parts();
        Ok(data.into_raw())
    }

    /// Copy the elements of a one-dimensional tensor into a raw buffer
    /// (`Vec<T>` for CPU devices).
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - `<B as DeviceRawAPI<T>>::Raw`: the gathered elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device));
    /// let v: Vec<i32> = a.to_vec();
    /// println!("{v:?}");
    /// // [0, 1, 2, 3, 4, 5]
    /// # assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the tensor is not one-dimensional.
    ///
    /// For a fallible version, use [`TensorAny::to_raw_f`].
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`Tensor::into_vec`]: consuming form returning `Vec<T>`.
    /// - [`asarray`](crate::tensor::asarray::asarray()): the inverse direction (buffer to tensor).
    pub fn to_vec(&self) -> <B as DeviceRawAPI<T>>::Raw {
        self.to_raw_f().rstsr_unwrap()
    }
}

impl<T, B, D> Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn into_raw_f(self) -> Result<<B as DeviceRawAPI<T>>::Raw> {
        rstsr_assert_eq!(self.ndim(), 1, InvalidLayout, "to_vec currently only support 1-D tensor")?;
        let layout = self.layout();
        let (idx_min, idx_max) = layout.bounds_index()?;
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == layout.size() && layout.stride()[0] > 0 {
            let (storage, _) = self.into_raw_parts();
            let (data, _) = storage.into_raw_parts();
            return Ok(data.into_raw());
        } else {
            return self.to_raw_f();
        }
    }

    pub fn into_raw(self) -> <B as DeviceRawAPI<T>>::Raw {
        self.into_raw_f().rstsr_unwrap()
    }
}

impl<T, B, D> Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
{
    pub fn into_vec_f(self) -> Result<Vec<T>> {
        rstsr_assert_eq!(self.ndim(), 1, InvalidLayout, "to_vec currently only support 1-D tensor")?;
        let layout = self.layout();
        let (idx_min, idx_max) = layout.bounds_index()?;
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == layout.size() && layout.stride()[0] > 0 {
            let (storage, _) = self.into_raw_parts();
            storage.into_cpu_vec()
        } else {
            let data = self.to_raw_f()?;
            let storage = Storage::new(DataOwned::from(data), self.device().clone());
            storage.into_cpu_vec()
        }
    }

    /// Convert a one-dimensional owned tensor into `Vec<T>`, moving the buffer
    /// when possible.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - `Vec<T>`: the elements, moved or gathered.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((6, &device));
    /// let v: Vec<i32> = a.into_vec();
    /// println!("{v:?}");
    /// // [0, 1, 2, 3, 4, 5]
    /// # assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the tensor is not one-dimensional.
    ///
    /// For a fallible version, use [`Tensor::into_vec_f`].
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`TensorAny::to_vec`]: borrowing form.
    /// - [`Tensor::into_raw`]: keep the device-specific raw buffer type.
    pub fn into_vec(self) -> Vec<T> {
        self.into_vec_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region to_scalar */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn to_scalar_f(&self) -> Result<T> {
        let layout = self.layout();
        rstsr_assert_eq!(layout.size(), 1, InvalidLayout)?;
        // Read the single element at the layout offset directly via `get_index`,
        // rather than materializing the whole buffer with `to_cpu_vec`. This avoids
        // a full storage clone for a one-element read and works for non-CPU devices
        // (which may not expose a cheap CPU `Vec`). Index 0 was previously read,
        // ignoring slicing/indexing offsets, so e.g. `arange(10).i(9)` (a 0-d view
        // at offset 9) wrongly returned 0 instead of 9.
        Ok(self.storage().get_index(layout.offset()))
    }

    /// Extract the single element of a size-one tensor as a scalar.
    ///
    /// The element is read at the layout offset, so a sliced 0-D view returns
    /// the value it points to, not the first buffer element.
    ///
    /// This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.
    ///
    /// # Returns
    ///
    /// - `T`: the scalar value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rstsr::prelude::*;
    /// # let mut device = DeviceCpu::default();
    /// # device.set_default_order(RowMajor);
    /// let a = rt::arange((10, &device));
    /// println!("{}", a.i(9).to_scalar());
    /// // 9
    /// # assert_eq!(a.i(9).to_scalar(), 9);
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the tensor does not have exactly one element (including the empty case).
    ///
    /// For a fallible version, use [`TensorAny::to_scalar_f`].
    ///
    /// # See also
    ///
    /// ## Related functions in RSTSR
    ///
    /// - [`Index`] operator `[]`: scalar access by index (boundary-checked).
    /// - [`TensorAny::to_vec`]: extract all elements of a 1-D tensor.
    pub fn to_scalar(&self) -> T {
        self.to_scalar_f().rstsr_unwrap()
    }
}

/* #endregion */

/* #region as_ptr */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    pub fn as_ptr(&self) -> *const T {
        unsafe { self.raw().as_ptr().add(self.layout().offset()) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T
    where
        R: DataMutAPI,
    {
        unsafe { self.raw_mut().as_mut_ptr().add(self.layout().offset()) }
    }
}

/* #endregion */

/* #region view API */

pub trait TensorViewAPI
where
    Self::Dim: DimAPI,
    Self::Backend: DeviceAPI<Self::Type>,
{
    type Type;
    type Backend;
    type Dim;
    /// Get a view of tensor.
    fn view(&self) -> TensorView<'_, Self::Type, Self::Backend, Self::Dim>;
}

impl<R, T, B, D> TensorViewAPI for TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view(&self) -> TensorView<'_, T, B, D> {
        let data = self.data().as_ref();
        let storage = Storage::new(data, self.device().clone());
        let layout = self.layout().clone();
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }
}

impl<R, T, B, D> TensorViewAPI for &TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view(&self) -> TensorView<'_, T, B, D> {
        TensorAny::view(*self)
    }
}

impl<R, T, B, D> TensorViewAPI for &mut TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view(&self) -> TensorView<'_, T, B, D> {
        TensorAny::view(*self)
    }
}

pub trait TensorViewMutAPI
where
    Self::Dim: DimAPI,
    Self::Backend: DeviceAPI<Self::Type>,
{
    type Type;
    type Backend;
    type Dim;

    /// Get a mutable view of tensor.
    fn view_mut(&mut self) -> TensorMut<'_, Self::Type, Self::Backend, Self::Dim>;
}

impl<R, T, B, D> TensorViewMutAPI for TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view_mut(&mut self) -> TensorMut<'_, T, B, D> {
        let device = self.device().clone();
        let layout = self.layout().clone();
        let data = self.data_mut().as_mut();
        let storage = Storage::new(data, device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }
}

impl<R, T, B, D> TensorViewMutAPI for &mut TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view_mut(&mut self) -> TensorMut<'_, T, B, D> {
        (*self).view_mut()
    }
}

pub trait TensorIntoOwnedAPI<T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    /// Convert tensor into owned tensor.
    ///
    /// Data is either moved or fully cloned.
    /// Layout is not involved; i.e. all underlying data is moved or cloned
    /// without changing layout.
    fn into_owned(self) -> Tensor<T, B, D>;
}

impl<R, T, B, D> TensorIntoOwnedAPI<T, B, D> for TensorAny<R, T, B, D>
where
    R: DataCloneAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    <B as DeviceRawAPI<T>>::Raw: Clone,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceRawAPI<MaybeUninit<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    fn into_owned(self) -> Tensor<T, B, D> {
        TensorAny::into_owned(self)
    }
}

/* #endregion */

/* #region tensor prop for computation */

pub trait TensorRefAPI<'l>: TensorViewAPI {}
impl<'l, R, T, B, D> TensorRefAPI<'l> for &'l TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    Self: TensorViewAPI,
{
}
impl<'l, T, B, D> TensorRefAPI<'l> for TensorView<'l, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    Self: TensorViewAPI,
{
}

pub trait TensorRefMutAPI<'l>: TensorViewAPI {}
impl<'l, R, T, B, D> TensorRefMutAPI<'l> for &mut TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    Self: TensorViewMutAPI,
{
}
impl<'l, T, B, D> TensorRefMutAPI<'l> for TensorMut<'l, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    Self: TensorViewMutAPI,
{
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_into_cow() {
        let mut a = arange(3);
        let ptr_a = a.raw().as_ptr();

        let a_mut = a.view_mut();
        let a_cow = a_mut.into_cow();
        println!("{a_cow:?}");

        let a_ref = a.view();
        let a_cow = a_ref.into_cow();
        println!("{a_cow:?}");

        let a_cow = a.into_cow();
        println!("{a_cow:?}");
        let ptr_a_cow = a_cow.raw().as_ptr();
        assert_eq!(ptr_a, ptr_a_cow);
    }

    #[test]
    #[ignore]
    fn test_force_mut() {
        let n = 4096;
        let a = linspace((0.0, 1.0, n * n)).into_shape((n, n));
        for _ in 0..10 {
            let time = std::time::Instant::now();
            for i in 0..n {
                let a_view = a.slice(i);
                let mut a_mut = unsafe { a_view.force_mut() };
                a_mut *= i as f64 / 2048.0;
            }
            println!("Elapsed time {:?}", time.elapsed());
        }
        println!("{a:16.10}");
    }

    #[test]
    #[ignore]
    #[cfg(feature = "rayon")]
    fn test_force_mut_par() {
        use rayon::prelude::*;
        let n = 4096;
        let a = linspace((0.0, 1.0, n * n)).into_shape((n, n));
        for _ in 0..10 {
            let time = std::time::Instant::now();
            (0..n).into_par_iter().for_each(|i| {
                let a_view = a.slice(i);
                let mut a_mut = unsafe { a_view.force_mut() };
                a_mut *= i as f64 / 2048.0;
            });
            println!("Elapsed time {:?}", time.elapsed());
        }
        println!("{a:16.10}");
    }
}
