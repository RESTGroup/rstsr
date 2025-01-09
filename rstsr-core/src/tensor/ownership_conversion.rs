use crate::prelude_dev::*;

/* #region basic conversion */

/// Methods for tensor ownership conversion.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    /// Get a view of tensor.
    pub fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        R: DataAPI,
    {
        let data = self.data().as_ref();
        let layout = self.layout().clone();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

    /// Get a mutable view of tensor.
    pub fn view_mut(&mut self) -> TensorBase<DataMut<'_, R::Data>, D>
    where
        R: DataMutAPI,
    {
        let layout = self.layout().clone();
        let data = self.data_mut().as_mut();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

    pub fn into_cow<'a>(self) -> TensorBase<DataCow<'a, R::Data>, D>
    where
        R: DataAPI + DataIntoCowAPI<'a>,
    {
        let (data, layout) = self.into_data_and_layout();
        let data = data.into_cow();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }

    /// Convert tensor into owned tensor.
    ///
    /// Data is either moved or cloned.
    /// Layout is not involved; i.e. all underlying data is moved or cloned
    /// without changing layout.
    ///
    /// # See also
    ///
    /// [`Tensor::into_owned`] keep data in some conditions, otherwise clone.
    /// This function can avoid cases where data memory bulk is large, but
    /// tensor view is small.
    pub fn into_owned_keep_layout(self) -> TensorBase<DataOwned<R::Data>, D>
    where
        R: DataAPI,
    {
        let TensorBase { data, layout } = self;
        let data = data.into_owned();
        unsafe { TensorBase::new_unchecked(data, layout) }
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
    pub fn into_shared_keep_layout(self) -> TensorBase<DataArc<R::Data>, D>
    where
        R: DataAPI,
    {
        let TensorBase { data, layout } = self;
        let data = data.into_shared();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    R::Data: Clone,
    D: DimAPI,
    T: Clone,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    pub fn into_owned(self) -> TensorBase<DataOwned<R::Data>, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().unwrap();
        if idx_min == 0 && idx_max == self.data().storage().len() && idx_max == self.layout().size()
        {
            return self.into_owned_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K));
        }
    }

    pub fn into_shared(self) -> TensorBase<DataArc<R::Data>, D> {
        let (idx_min, idx_max) = self.layout().bounds_index().unwrap();
        if idx_min == 0 && idx_max == self.data().storage().len() && idx_max == self.layout().size()
        {
            return self.into_shared_keep_layout();
        } else {
            return asarray((&self, TensorIterOrder::K)).into_shared();
        }
    }

    pub fn to_owned(&self) -> TensorBase<DataOwned<R::Data>, D> {
        self.view().into_owned()
    }
}

/* #endregion */

/* #region to_vector */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn to_vec_f(&self) -> Result<Vec<T>> {
        rstsr_assert_eq!(
            self.ndim(),
            1,
            InvalidLayout,
            "to_vec currently only support 1-D tensor"
        )?;
        let data = self.data().storage();
        let layout = self.layout().to_dim::<Ix1>()?;
        let device = data.device();
        let size = layout.size();
        let mut new_data = unsafe { device.empty_impl(size)? };
        device.assign(&mut new_data, &[size].c(), data, &layout)?;
        Ok(new_data.into_rawvec())
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.to_vec_f().unwrap()
    }
}

impl<T, D, B> Tensor<T, D, B>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    pub fn into_vec_f(self) -> Result<Vec<T>> {
        rstsr_assert_eq!(
            self.ndim(),
            1,
            InvalidLayout,
            "to_vec currently only support 1-D tensor"
        )?;
        let layout = self.layout();
        let (idx_min, idx_max) = layout.bounds_index()?;
        if idx_min == 0
            && idx_max == self.data.storage.len()
            && idx_max == layout.size()
            && layout.stride()[0] > 0
        {
            return Ok(self.data.storage.into_rawvec());
        } else {
            return self.to_vec_f();
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.into_vec_f().unwrap()
    }
}

impl<T, D, B> Tensor<T, D, B>
where
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>>,
{
    pub fn into_rawvec(self) -> B::RawVec {
        self.data.storage.into_rawvec()
    }
}

/* #endregion */

/* #region to_scalar */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn to_scalar_f(&self) -> Result<T> {
        let layout = self.layout();
        rstsr_assert_eq!(layout.size(), 1, InvalidLayout)?;
        let data = self.data().storage();
        let vec = data.to_cpu_vec()?;
        Ok(vec[0].clone())
    }

    pub fn to_scalar(&self) -> T {
        self.to_scalar_f().unwrap()
    }
}

/* #endregion */

/* #region as_ptr */

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T, RawVec = Vec<T>>,
{
    pub fn as_ptr(&self) -> *const T {
        unsafe { self.rawvec().as_ptr().add(self.layout().offset()) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T
    where
        R: DataMutAPI,
    {
        unsafe { self.rawvec_mut().as_mut_ptr().add(self.layout().offset()) }
    }
}

/* #endregion */

/* #region view API */

pub trait TensorViewAPI<S, D>
where
    D: DimAPI,
{
    /// Get a view of tensor.
    fn view(&self) -> TensorBase<DataRef<'_, S>, D>;
}

impl<R, D> TensorViewAPI<R::Data, D> for TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D> {
        let data = self.data().as_ref();
        let layout = self.layout().clone();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<R, D> TensorViewAPI<R::Data, D> for &TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    fn view(&self) -> TensorBase<DataRef<'_, R::Data>, D> {
        (*self).view()
    }
}

pub trait TensorViewMutAPI<S, D>
where
    D: DimAPI,
{
    /// Get a mutable view of tensor.
    fn view_mut(&mut self) -> TensorBase<DataMut<'_, S>, D>;
}

impl<R, D> TensorViewMutAPI<R::Data, D> for TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
{
    fn view_mut(&mut self) -> TensorBase<DataMut<'_, R::Data>, D> {
        let layout = self.layout().clone();
        let data = self.data_mut().as_mut();
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<R, D> TensorViewMutAPI<R::Data, D> for &mut TensorBase<R, D>
where
    R: DataMutAPI,
    D: DimAPI,
{
    fn view_mut(&mut self) -> TensorBase<DataMut<'_, R::Data>, D> {
        (*self).view_mut()
    }
}

/* #endregion */

/* #region tensor prop for computation */

pub trait TensorRefAPI {}
impl<R, D> TensorRefAPI for &TensorBase<R, D>
where
    D: DimAPI,
    R: DataAPI,
    Self: TensorViewAPI<R::Data, D>,
{
}
impl<S, D> TensorRefAPI for TensorBase<DataRef<'_, S>, D>
where
    D: DimAPI,
    Self: TensorViewAPI<S, D>,
{
}

pub trait TensorRefMutAPI {}
impl<R, D> TensorRefMutAPI for &mut TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    Self: TensorViewMutAPI<R::Data, D>,
{
}
impl<S, D> TensorRefMutAPI for TensorBase<DataMut<'_, S>, D>
where
    D: DimAPI,
    Self: TensorViewMutAPI<S, D>,
{
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_into_cow() {
        let mut a = arange(3);
        let ptr_a = a.rawvec().as_ptr();

        let a_mut = a.view_mut();
        let a_cow = a_mut.into_cow();
        println!("{:?}", a_cow);

        let a_ref = a.view();
        let a_cow = a_ref.into_cow();
        println!("{:?}", a_cow);

        let a_cow = a.into_cow();
        println!("{:?}", a_cow);
        let ptr_a_cow = a_cow.rawvec().as_ptr();
        assert_eq!(ptr_a, ptr_a_cow);
    }
}
