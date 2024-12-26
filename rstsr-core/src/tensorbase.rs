use crate::prelude_dev::*;

pub trait TensorBaseAPI {}

#[derive(Clone)]
pub struct TensorBase<R, D>
where
    D: DimAPI,
{
    pub(crate) data: R,
    pub(crate) layout: Layout<D>,
}

impl<R, D> TensorBaseAPI for TensorBase<R, D> where D: DimAPI {}

/// Basic definitions for tensor object.
impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    /// Initialize tensor object.
    ///
    /// # Safety
    ///
    /// This function will not check whether data meets the standard of
    /// [Storage<T, B>], or whether layout may exceed pointer bounds of data.
    pub unsafe fn new_unchecked(data: R, layout: Layout<D>) -> Self {
        Self { data, layout }
    }

    #[inline]
    pub fn data(&self) -> &R {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut R {
        &mut self.data
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
}

impl<T, D, B, R> TensorBase<R, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    R: DataAPI<Data = Storage<T, B>>,
{
    pub fn new_f(data: R, layout: Layout<D>) -> Result<Self> {
        // check stride sanity
        layout.check_strides()?;

        // check pointer exceed
        let len_data = data.storage().len();
        let (_, idx_max) = layout.bounds_index()?;
        rstsr_pattern!(idx_max, len_data.., ValueOutOfRange)?;
        return Ok(Self { data, layout });
    }

    pub fn new(data: R, layout: Layout<D>) -> Self {
        Self::new_f(data, layout).unwrap()
    }

    pub fn device<'a>(&'a self) -> &'a B
    where
        T: 'a,
    {
        self.data().storage().device()
    }

    pub fn storage(&self) -> &Storage<T, B> {
        self.data().storage()
    }
}

impl<T, D, B, R> TensorBase<R, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
    R: DataMutAPI<Data = Storage<T, B>>,
{
    #[inline]
    pub fn storage_mut(&mut self) -> &mut Storage<T, B> {
        self.data_mut().storage_mut()
    }
}

impl<R, D> TensorBase<R, D>
where
    D: DimAPI,
{
    pub fn into_data(self) -> R {
        self.data
    }

    pub fn into_data_and_layout(self) -> (R, Layout<D>) {
        (self.data, self.layout)
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

pub type Tensor<T, D, B = DeviceCpu> = TensorBase<DataOwned<Storage<T, B>>, D>;
pub type TensorView<'a, T, D, B = DeviceCpu> = TensorBase<DataRef<'a, Storage<T, B>>, D>;
pub type TensorViewMut<'a, T, D, B = DeviceCpu> = TensorBase<DataMut<'a, Storage<T, B>>, D>;
pub type TensorCow<'a, T, D, B = DeviceCpu> = TensorBase<DataCow<'a, Storage<T, B>>, D>;
pub type TensorArc<T, D, B = DeviceCpu> = TensorBase<DataArc<Storage<T, B>>, D>;
pub use TensorView as TensorRef;
pub use TensorViewMut as TensorMut;
