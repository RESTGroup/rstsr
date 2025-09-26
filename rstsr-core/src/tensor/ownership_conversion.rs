use crate::prelude_dev::*;

/* #region basic conversion */

/// Methods for tensor ownership conversion.
impl<R, T, B, D> TensorAny<R, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    /// Get a view of tensor.
    pub fn view(&self) -> TensorView<'_, T, B, D> {
        let layout = self.layout().clone();
        let data = self.data().as_ref();
        let storage = Storage::new(data, self.device().clone());
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Get a mutable view of tensor.
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

    /// Convert current tensor into copy-on-write.
    pub fn into_cow<'a>(self) -> TensorCow<'a, T, B, D>
    where
        R: DataIntoCowAPI<'a>,
    {
        let (storage, layout) = self.into_raw_parts();
        let (data, device) = storage.into_raw_parts();
        let storage = Storage::new(data.into_cow(), device);
        unsafe { TensorBase::new_unchecked(storage, layout) }
    }

    /// Convert tensor into owned tensor.
    ///
    /// Data is either moved or fully cloned.
    /// Layout is not involved; i.e. all underlying data is moved or cloned
    /// without changing layout.
    ///
    /// # See also
    ///
    /// [`Tensor::into_owned`] keep data in some conditions, otherwise clone.
    /// This function can avoid cases where data memory bulk is large, but
    /// tensor view is small.
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

    /// Convert tensor into shared tensor.
    ///
    /// Data is either moved or cloned.
    /// Layout is not involved; i.e. all underlying data is moved or cloned
    /// without changing layout.
    ///
    /// # See also
    ///
    /// [`Tensor::into_shared`] keep data in some conditions, otherwise clone.
    /// This function can avoid cases where data memory bulk is large, but
    /// tensor view is small.
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
    pub fn into_owned(self) -> Tensor<T, B, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().unwrap();
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == self.layout().size() {
            return self.into_owned_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K));
        }
    }

    pub fn into_shared(self) -> TensorArc<T, B, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().unwrap();
        if idx_min == 0 && idx_max == self.storage().len() && idx_max == self.layout().size() {
            return self.into_shared_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K)).into_shared();
        }
    }

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
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn to_raw_f(&self) -> Result<Vec<T>> {
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

    pub fn to_vec(&self) -> Vec<T> {
        self.to_raw_f().unwrap()
    }
}

impl<T, B, D> Tensor<T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn into_vec_f(self) -> Result<Vec<T>> {
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

    pub fn into_vec(self) -> Vec<T> {
        self.into_vec_f().unwrap()
    }
}

/* #endregion */

/* #region to_scalar */

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataCloneAPI<Data = B::Raw>,
    B::Raw: Clone,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn to_scalar_f(&self) -> Result<T> {
        let layout = self.layout();
        rstsr_assert_eq!(layout.size(), 1, InvalidLayout)?;
        let storage = self.storage();
        let vec = storage.to_cpu_vec()?;
        Ok(vec[0].clone())
    }

    pub fn to_scalar(&self) -> T {
        self.to_scalar_f().unwrap()
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
        (*self).view()
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

pub trait TensorRefAPI {}
impl<R, T, B, D> TensorRefAPI for &TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    Self: TensorViewAPI,
{
}
impl<T, B, D> TensorRefAPI for TensorView<'_, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    Self: TensorViewAPI,
{
}

pub trait TensorRefMutAPI {}
impl<R, T, B, D> TensorRefMutAPI for &mut TensorAny<R, T, B, D>
where
    D: DimAPI,
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    Self: TensorViewMutAPI,
{
}
impl<T, B, D> TensorRefMutAPI for TensorMut<'_, T, B, D>
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
