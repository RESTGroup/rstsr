use crate::prelude_dev::*;

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
        Self: Into<TensorBase<DataCow<'a, R::Data>, D>>,
        R: DataAPI,
    {
        self.into()
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

/* #region DataCow */

impl<S, D> From<TensorBase<DataOwned<S>, D>> for TensorBase<DataCow<'_, S>, D>
where
    D: DimAPI,
{
    #[inline]
    fn from(tensor: TensorBase<DataOwned<S>, D>) -> Self {
        let (data, layout) = tensor.into_data_and_layout();
        let data = DataCow::from(data);
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<'a, S, D> From<TensorBase<DataRef<'a, S>, D>> for TensorBase<DataCow<'a, S>, D>
where
    D: DimAPI,
{
    #[inline]
    fn from(tensor: TensorBase<DataRef<'a, S>, D>) -> Self {
        let (data, layout) = tensor.into_data_and_layout();
        let data = DataCow::from(data);
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

impl<'a, S, D> From<TensorBase<DataMut<'a, S>, D>> for TensorBase<DataCow<'a, S>, D>
where
    D: DimAPI,
    S: Clone,
{
    #[inline]
    fn from(tensor: TensorBase<DataMut<'a, S>, D>) -> Self {
        let (data, layout) = tensor.into_data_and_layout();
        match data {
            DataMut::TrueRef(data) => {
                let data = DataCow::from(DataRef::from(&*data));
                unsafe { TensorBase::new_unchecked(data, layout) }
            },
            DataMut::ManuallyDropOwned(data) => {
                let data = DataCow::from(DataRef::from_manually_drop(data));
                unsafe { TensorBase::new_unchecked(data, layout) }
            },
        }
    }
}

impl<S, D> From<TensorBase<DataArc<S>, D>> for TensorBase<DataCow<'_, S>, D>
where
    D: DimAPI,
    S: Clone,
{
    #[inline]
    fn from(tensor: TensorBase<DataArc<S>, D>) -> Self {
        let (data, layout) = tensor.into_data_and_layout();
        let data = data.into_owned();
        let data = DataCow::from(data);
        unsafe { TensorBase::new_unchecked(data, layout) }
    }
}

/* #endregion */

/* #region operation API */

/// This trait is used for implementing operations that involves view-only
/// input.
pub trait TensorRefAPI<S, D>
where
    D: DimAPI,
{
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D>;
}

impl<R, S, D> TensorRefAPI<S, D> for &TensorBase<R, D>
where
    R: DataAPI<Data = S>,
    D: DimAPI,
{
    #[inline]
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D> {
        self.view()
    }
}

impl<S, D> TensorRefAPI<S, D> for TensorBase<DataRef<'_, S>, D>
where
    S: Clone,
    D: DimAPI,
{
    #[inline]
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D> {
        self.view()
    }
}

/// This trait is used for implementing operations that involves view-only
/// operation, but input can be view-only or owned.
pub trait TensorRefOrOwnedAPI<S, D>
where
    D: DimAPI,
{
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D>;
}

impl<R, S, D> TensorRefOrOwnedAPI<S, D> for &TensorBase<R, D>
where
    R: DataAPI<Data = S>,
    D: DimAPI,
{
    #[inline]
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D> {
        self.view()
    }
}

impl<S, D> TensorRefOrOwnedAPI<S, D> for TensorBase<DataRef<'_, S>, D>
where
    S: Clone,
    D: DimAPI,
{
    #[inline]
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D> {
        self.view()
    }
}

impl<S, D> TensorRefOrOwnedAPI<S, D> for TensorBase<DataOwned<S>, D>
where
    S: Clone,
    D: DimAPI,
{
    #[inline]
    fn tsr_view(&self) -> TensorBase<DataRef<'_, S>, D> {
        self.view()
    }
}

/// This trait is used for implementing operations that involves view-mut
/// input.
pub trait TensorRefMutAPI<S, D>
where
    D: DimAPI,
{
    fn tsr_view_mut(&mut self) -> TensorBase<DataMut<'_, S>, D>;
}

impl<R, S, D> TensorRefMutAPI<S, D> for &mut TensorBase<R, D>
where
    R: DataMutAPI<Data = S>,
    D: DimAPI,
{
    #[inline]
    fn tsr_view_mut(&mut self) -> TensorBase<DataMut<'_, S>, D> {
        self.view_mut()
    }
}

impl<S, D> TensorRefMutAPI<S, D> for TensorBase<DataMut<'_, S>, D>
where
    S: Clone,
    D: DimAPI,
{
    #[inline]
    fn tsr_view_mut(&mut self) -> TensorBase<DataMut<'_, S>, D> {
        self.view_mut()
    }
}

/* #endregion */
