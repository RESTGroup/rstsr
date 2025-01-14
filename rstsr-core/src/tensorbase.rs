use crate::prelude_dev::*;

pub trait TensorBaseAPI {}

#[derive(Clone)]
pub struct TensorBase<S, D>
where
    D: DimAPI,
{
    pub(crate) storage: S,
    pub(crate) layout: Layout<D>,
}

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
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    D: DimAPI,
    B: DeviceAPI<T>,
{
    pub fn new_f(storage: Storage<R, T, B>, layout: Layout<D>) -> Result<Self> {
        // check stride sanity
        layout.check_strides()?;

        // check pointer exceed
        let len_data = storage.len();
        let (_, idx_max) = layout.bounds_index()?;
        rstsr_pattern!(idx_max, ..=len_data, ValueOutOfRange)?;
        return Ok(Self { storage, layout });
    }

    pub fn new(storage: Storage<R, T, B>, layout: Layout<D>) -> Self {
        Self::new_f(storage, layout).unwrap()
    }

    pub fn device(&self) -> &B {
        self.storage().device()
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

pub type Tensor<T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataOwned<<B as DeviceRawAPI<T>>::Raw>, T, B>, D>;
pub type TensorView<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataRef<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;
pub type TensorViewMut<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataMut<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;
pub type TensorCow<'a, T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataCow<'a, <B as DeviceRawAPI<T>>::Raw>, T, B>, D>;
pub type TensorArc<T, B = DeviceCpu, D = IxD> =
    TensorBase<Storage<DataArc<<B as DeviceRawAPI<T>>::Raw>, T, B>, D>;
pub type TensorAny<R, T, B, D> = TensorBase<Storage<R, T, B>, D>;
pub use TensorView as TensorRef;
pub use TensorViewMut as TensorMut;
