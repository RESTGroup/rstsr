use crate::prelude_dev::*;

/* #region into_compatible_shape */

pub fn into_compatible_shape_f<R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error = Error>,
    order: impl Into<Option<FlagOrder>>,
) -> Result<TensorAny<R, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    let shape_new = reshape_substitute_negatives(shape.try_into()?.as_ref(), tensor.size())?;
    let order = order.into().unwrap_or(tensor.device().default_order());
    if let Some(layout_new) = layout_reshapeable(&tensor.layout().to_dim()?, &shape_new, order)? {
        let (storage, _) = tensor.into_raw_parts();
        unsafe { Ok(TensorBase::new_unchecked(storage, layout_new)) }
    } else {
        rstsr_raise!(InvalidLayout, "Cannot reshape {:?} to {shape_new:?} with order {order:?}.", tensor.layout())?
    }
}

pub fn into_compatible_shape<R, T, B, D>(
    tensor: TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error = Error>,
    order: impl Into<Option<FlagOrder>>,
) -> TensorAny<R, T, B, IxD>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_compatible_shape_f(tensor, shape, order).rstsr_unwrap()
}

pub fn to_compatible_shape_f<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error = Error>,
    order: impl Into<Option<FlagOrder>>,
) -> Result<TensorView<'_, T, B, IxD>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    into_compatible_shape_f(tensor.view(), shape, order)
}

pub fn to_compatible_shape<R, T, B, D>(
    tensor: &TensorAny<R, T, B, D>,
    shape: impl TryInto<AxesIndex<isize>, Error = Error>,
    order: impl Into<Option<FlagOrder>>,
) -> TensorView<'_, T, B, IxD>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    to_compatible_shape_f(tensor, shape, order).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn into_compatible_shape_f(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error = Error>,
        order: impl Into<Option<FlagOrder>>,
    ) -> Result<TensorAny<R, T, B, IxD>> {
        into_compatible_shape_f(self, shape, order)
    }

    pub fn into_compatible_shape(
        self,
        shape: impl TryInto<AxesIndex<isize>, Error = Error>,
        order: impl Into<Option<FlagOrder>>,
    ) -> TensorAny<R, T, B, IxD> {
        into_compatible_shape(self, shape, order)
    }

    pub fn to_compatible_shape_f(
        &self,
        shape: impl TryInto<AxesIndex<isize>, Error = Error>,
        order: impl Into<Option<FlagOrder>>,
    ) -> Result<TensorView<'_, T, B, IxD>> {
        to_compatible_shape_f(self, shape, order)
    }

    pub fn to_compatible_shape(
        &self,
        shape: impl TryInto<AxesIndex<isize>, Error = Error>,
        order: impl Into<Option<FlagOrder>>,
    ) -> TensorView<'_, T, B, IxD> {
        to_compatible_shape(self, shape, order)
    }
}

/* #endregion */

/* #region reshape_assume_contig (deprecated) */

pub fn into_shape_assume_contig_f<R, T, B, D, D2>(
    tensor: TensorAny<R, T, B, D>,
    shape: D2,
) -> Result<TensorAny<R, T, B, D2>>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
    D2: DimAPI,
{
    let default_order = tensor.device().default_order();
    let (storage, layout) = tensor.into_raw_parts();

    rstsr_assert_eq!(layout.size(), shape.shape_size(), InvalidLayout, "Number of elements not same.")?;

    let new_layout = {
        if default_order == FlagOrder::C && layout.c_contig() {
            shape.new_c_contig(Some(layout.offset()))
        } else if default_order == FlagOrder::F && layout.f_contig() {
            shape.new_f_contig(Some(layout.offset()))
        } else {
            rstsr_raise!(InvalidLayout, "This array is not contiguous by {:?}", default_order)?
        }
    };
    unsafe { Ok(TensorBase::new_unchecked(storage, new_layout)) }
}

/// Assuming contiguous array, reshapes an array without changing its data.
///
/// This function may return c-contiguous or f-contiguous array depending on
/// crate feature `f_prefer`.
///
/// # See also
///
/// [Python array API standard: `reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html)
pub fn to_shape_assume_contig<R, T, B, D, D2>(tensor: &TensorAny<R, T, B, D>, shape: D2) -> TensorView<'_, T, B, D2>
where
    D: DimAPI,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_shape_assume_contig_f(tensor.view(), shape).rstsr_unwrap()
}

pub fn to_shape_assume_contig_f<R, T, B, D, D2>(
    tensor: &TensorAny<R, T, B, D>,
    shape: D2,
) -> Result<TensorView<'_, T, B, D2>>
where
    D: DimAPI,
    D2: DimAPI,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_shape_assume_contig_f(tensor.view(), shape)
}

pub fn into_shape_assume_contig<R, T, B, D, D2>(tensor: TensorAny<R, T, B, D>, shape: D2) -> TensorAny<R, T, B, D2>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
    D2: DimAPI,
{
    into_shape_assume_contig_f(tensor, shape).rstsr_unwrap()
}

pub use to_shape_assume_contig as reshape_assume_contig;
pub use to_shape_assume_contig_f as reshape_assume_contig_f;

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// # See also
    ///
    /// [`reshape_assume_contig`]
    pub fn reshape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    pub fn reshape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    pub fn to_shape_assume_contig<D2>(&self, shape: D2) -> TensorView<'_, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self.view(), shape)
    }

    pub fn to_shape_assume_contig_f<D2>(&self, shape: D2) -> Result<TensorView<'_, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self.view(), shape)
    }

    pub fn into_shape_assume_contig<D2>(self, shape: D2) -> TensorAny<R, T, B, D2>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig(self, shape)
    }

    pub fn into_shape_assume_contig_f<D2>(self, shape: D2) -> Result<TensorAny<R, T, B, D2>>
    where
        D2: DimAPI,
    {
        into_shape_assume_contig_f(self, shape)
    }
}

/* #endregion */
