//! This module handles tensor data manipulation.

use crate::prelude_dev::*;
use core::num::TryFromIntError;

/* #region broadcast_arrays */

/// Broadcasts any number of arrays against each other.
///
/// # See also
///
/// [Python Array API standard: `broadcast_arrays`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_arrays.html)
pub fn broadcast_arrays<R>(tensors: Vec<TensorBase<R, IxD>>) -> Vec<TensorBase<R, IxD>>
where
    R: DataAPI,
{
    broadcast_arrays_f(tensors).unwrap()
}

pub fn broadcast_arrays_f<R>(tensors: Vec<TensorBase<R, IxD>>) -> Result<Vec<TensorBase<R, IxD>>>
where
    R: DataAPI,
{
    // fast return if there is only zero/one tensor
    if tensors.len() <= 1 {
        return Ok(tensors);
    }
    let mut shape_b = tensors[0].shape().clone();
    for tensor in tensors.iter().skip(1) {
        let shape = tensor.shape();
        let (shape, _, _) = broadcast_shape(shape, &shape_b)?;
        shape_b = shape;
    }
    let mut tensors_new = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let tensor = into_broadcast_f(tensor, shape_b.clone())?;
        tensors_new.push(tensor);
    }
    return Ok(tensors_new);
}

/* #endregion */

/* #region broadcast_to */

pub fn into_broadcast_f<R, D, D2>(tensor: TensorBase<R, D>, shape: D2) -> Result<TensorBase<R, D2>>
where
    R: DataAPI,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    let shape1 = tensor.shape();
    let shape2 = &shape;
    let (shape, tp1, _) = broadcast_shape(shape1, shape2)?;
    let layout = update_layout_by_shape(tensor.layout(), &shape, &tp1)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// [Python Array API standard: `broadcast_to`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_to.html)
pub fn to_broadcast<R, D, D2>(
    tensor: &TensorBase<R, D>,
    shape: D2,
) -> TensorBase<DataRef<'_, R::Data>, D2>
where
    R: DataAPI,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    into_broadcast_f(tensor.view(), shape).unwrap()
}

pub fn to_broadcast_f<R, D, D2>(
    tensor: &TensorBase<R, D>,
    shape: D2,
) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
where
    R: DataAPI,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    into_broadcast_f(tensor.view(), shape)
}

pub fn into_broadcast<R, D, D2>(tensor: TensorBase<R, D>, shape: D2) -> TensorBase<R, D2>
where
    R: DataAPI,
    D: DimAPI + DimMaxAPI<D2, Max = D2>,
    D2: DimAPI,
{
    into_broadcast_f(tensor, shape).unwrap()
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// [`broadcast_to`]
    pub fn to_broadcast<D2>(&self, shape: D2) -> TensorBase<DataRef<'_, R::Data>, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        to_broadcast(self, shape)
    }

    pub fn to_broadcast_f<D2>(&self, shape: D2) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        to_broadcast_f(self, shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// [`broadcast_to`]
    pub fn into_broadcast<D2>(self, shape: D2) -> TensorBase<R, D2>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast(self, shape)
    }

    pub fn into_broadcast_f<D2>(self, shape: D2) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        into_broadcast_f(self, shape)
    }
}

/* #endregion */

/* #region expand_dims */

pub fn into_expand_dims_f<I, R, D>(
    tensor: TensorBase<R, D>,
    axis: I,
) -> Result<TensorBase<R, D::LargerOne>>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    I: TryInto<isize>,
{
    let axis = axis.try_into().map_err(|_| Error::TryFromIntError(String::new()))?;
    let layout = tensor.layout().dim_insert(axis)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

/// Expands the shape of an array by inserting a new axis (dimension) of size
/// one at the position specified by `axis`.
///
/// # Panics
///
/// - If `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// [Python Array API standard: `expand_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.expand_dims.html)
pub fn expand_dims<I, R, D>(
    tensor: &TensorBase<R, D>,
    axis: I,
) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    I: TryInto<isize>,
{
    into_expand_dims_f(tensor.view(), axis).unwrap()
}

pub fn expand_dims_f<I, R, D>(
    tensor: &TensorBase<R, D>,
    axis: I,
) -> Result<TensorBase<DataRef<'_, R::Data>, D::LargerOne>>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    I: TryInto<isize>,
{
    into_expand_dims_f(tensor.view(), axis)
}

pub fn into_expand_dims<I, R, D>(tensor: TensorBase<R, D>, axis: I) -> TensorBase<R, D::LargerOne>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    I: TryInto<isize>,
{
    into_expand_dims_f(tensor, axis).unwrap()
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
{
    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn expand_dims<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
    where
        I: TryInto<isize>,
    {
        into_expand_dims(self.view(), axis)
    }

    pub fn expand_dims_f<I>(
        &self,
        axis: I,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D::LargerOne>>
    where
        I: TryInto<isize>,
    {
        into_expand_dims_f(self.view(), axis)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn into_expand_dims<I>(self, axis: I) -> TensorBase<R, D::LargerOne>
    where
        I: TryInto<isize>,
    {
        into_expand_dims(self, axis)
    }

    pub fn into_expand_dims_f<I>(self, axis: I) -> Result<TensorBase<R, D::LargerOne>>
    where
        I: TryInto<isize>,
    {
        into_expand_dims_f(self, axis)
    }
}

/* #endregion */

/* #region flip */

pub fn into_flip_f<I, R, D>(tensor: TensorBase<R, D>, axes: I) -> Result<TensorBase<R, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    let mut layout = tensor.layout().clone();
    let axes = axes.try_into()?;
    match axes {
        AxesIndex::Val(axis) => {
            layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
        },
        AxesIndex::Vec(axes) => {
            for &axis in axes.iter() {
                layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
            }
        },
    }
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

/// Reverses the order of elements in an array along the given axis.
///
/// # Panics
///
/// - If some index in `axis` is greater than the number of axes in the original
///   tensor.
///
/// # See also
///
/// [Python array API standard: `flip`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.flip.html)
pub fn flip<I, R, D>(tensor: &TensorBase<R, D>, axes: I) -> TensorBase<DataRef<'_, R::Data>, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_flip_f(tensor.view(), axes).unwrap()
}

pub fn flip_f<I, R, D>(
    tensor: &TensorBase<R, D>,
    axes: I,
) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_flip_f(tensor.view(), axes)
}

pub fn into_flip<I, R, D>(tensor: TensorBase<R, D>, axes: I) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_flip_f(tensor, axes).unwrap()
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn flip<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        flip(self, axis)
    }

    pub fn flip_f<I>(&self, axis: I) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        flip_f(self, axis)
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn into_flip<I>(self, axis: I) -> TensorBase<R, D>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip(self, axis)
    }

    pub fn into_flip_f<I>(self, axis: I) -> Result<TensorBase<R, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip_f(self, axis)
    }
}

/* #endregion */

/* #region permute_dims */

/// Permutes the axes (dimensions) of an array `x`.
///
/// # See also
///
/// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
pub fn transpose<I, R, D>(tensor: TensorBase<R, D>, axes: &[I]) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    transpose_f(tensor, axes).unwrap()
}

pub fn transpose_f<I, R, D>(tensor: TensorBase<R, D>, axes: &[I]) -> Result<TensorBase<R, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError> + Copy,
{
    let axes: Vec<isize> = axes.iter().map(|&x| x.try_into()).try_collect()?;
    let layout = tensor.layout().transpose(&axes)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

pub use transpose_f as permute_dims;

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn transpose<I>(&self, axes: &[I]) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose(self.view(), axes)
    }

    pub fn transpose_f<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose_f(self.view(), axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_transpose<I>(self, axes: &[I]) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose(self, axes)
    }

    pub fn into_transpose_f<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose_f(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn permute_dims<I>(&self, axes: &[I]) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose(self.view(), axes)
    }

    pub fn permute_dims_f<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose_f(self.view(), axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_permute_dims<I>(self, axes: &[I]) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose(self, axes)
    }

    pub fn into_permute_dims_f<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError> + Copy,
    {
        transpose_f(self, axes)
    }
}

/* #endregion */

/* #region reverse_axes */

/// Reverse the order of elements in an array along the given axis.
pub fn reverse_axes<R, D>(tensor: TensorBase<R, D>) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    let layout = tensor.layout().reverse_axes();
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`reverse_axes`]
    pub fn reverse_axes(&self) -> TensorBase<DataRef<'_, R::Data>, D> {
        reverse_axes(self.view())
    }

    /// Reverse the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`reverse_axes`]
    pub fn into_reverse_axes(self) -> TensorBase<R, D> {
        reverse_axes(self)
    }
}

/* #endregion */

/* #region swapaxes */

/// Interchange two axes of an array.
///
/// # See also
///
/// - [numpy `swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html)
pub fn swapaxes<I, R, D>(tensor: TensorBase<R, D>, axis1: I, axis2: I) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    swapaxes_f(tensor, axis1, axis2).unwrap()
}

pub fn swapaxes_f<I, R, D>(tensor: TensorBase<R, D>, axis1: I, axis2: I) -> Result<TensorBase<R, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis1 = axis1.try_into()?;
    let axis2 = axis2.try_into()?;
    let layout = tensor.layout().swapaxes(axis1, axis2)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`swapaxes`]
    pub fn swapaxes<I>(&self, axis1: I, axis2: I) -> TensorBase<DataRef<'_, R::Data>, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        swapaxes(self.view(), axis1, axis2)
    }

    pub fn swapaxes_f<I>(&self, axis1: I, axis2: I) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        swapaxes_f(self.view(), axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`swapaxes`]
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorBase<R, D>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        swapaxes(self, axis1, axis2)
    }

    pub fn into_swapaxes_f<I>(self, axis1: I, axis2: I) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        swapaxes_f(self, axis1, axis2)
    }
}

/* #endregion */

/* #region squeeze */

/// Removes singleton dimensions (axes) from `x`.
///
/// # See also
///
/// [Python array API standard: `squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html)
pub fn squeeze<I, R, D>(tensor: TensorBase<R, D>, axis: I) -> TensorBase<R, D::SmallerOne>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    squeeze_f(tensor, axis).unwrap()
}

pub fn squeeze_f<I, R, D>(tensor: TensorBase<R, D>, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into()?;
    let layout = tensor.layout().dim_eliminate(axis)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn squeeze<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D::SmallerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        squeeze(self.view(), axis)
    }

    pub fn squeeze_f<I>(&self, axis: I) -> Result<TensorBase<DataRef<'_, R::Data>, D::SmallerOne>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        squeeze_f(self.view(), axis)
    }

    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn into_squeeze<I>(self, axis: I) -> TensorBase<R, D::SmallerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        squeeze(self, axis)
    }

    pub fn into_squeeze_f<I>(self, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        squeeze_f(self, axis)
    }
}

/* #endregion */

/* #region into_dim */

/// Convert layout to another dimension.
///
/// This is mostly used when converting static dimension to dynamic
/// dimension or vice versa.
pub fn into_dim<R, D, D2>(tensor: TensorBase<R, D>) -> Result<TensorBase<R, D2>>
where
    R: DataAPI,
    D: DimAPI,
    D2: DimAPI,
    D: DimIntoAPI<D2>,
{
    let layout = tensor.layout().clone().into_dim::<D2>()?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Convert layout to another dimension.
    ///
    /// This is mostly used when converting static dimension to dynamic
    /// dimension or vice versa.
    ///
    /// # See also
    ///
    /// [`into_dim`]
    pub fn to_dim<D2>(&self) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim(self.view())
    }

    /// Convert layout to another dimension.
    ///
    /// # See also
    ///
    /// [`into_dim`]
    pub fn into_dim<D2>(self) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        D: DimIntoAPI<D2>,
    {
        into_dim(self)
    }
}

/* #endregion */

/* #region reshape_assume_contig */

/// Assuming contiguous array, reshapes an array without changing its data.
///
/// This function may return c-contiguous or f-contiguous array:
/// - If input array is both c-contiguous and f-contiguous (especially case of
///   1-D), the output array will be chosen as default contiguous.
/// - If input array is c-contiguous but not f-contiguous, the output array will
///   be c-contiguous.
/// - If input array is f-contiguous but not c-contiguous, the output array will
///   be f-contiguous.
///
/// # See also
///
/// [Python array API standard: `reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html)
pub fn reshape_assume_contig<R, D, D2>(tensor: TensorBase<R, D>, shape: D2) -> TensorBase<R, D2>
where
    R: DataAPI,
    D: DimAPI,
    D2: DimAPI,
{
    reshape_assume_contig_f(tensor, shape).unwrap()
}

pub fn reshape_assume_contig_f<R, D, D2>(
    tensor: TensorBase<R, D>,
    shape: D2,
) -> Result<TensorBase<R, D2>>
where
    R: DataAPI,
    D: DimAPI,
    D2: DimAPI,
{
    let layout = tensor.layout();
    let is_c_contig = layout.c_contig();
    let is_f_contig = layout.f_contig();

    rstsr_assert_eq!(
        layout.size(),
        shape.shape_size(),
        InvalidLayout,
        "Number of elements not same."
    )?;

    let new_layout = match (is_c_contig, is_f_contig) {
        (true, true) => match TensorOrder::default() {
            TensorOrder::C => shape.new_c_contig(Some(layout.offset)),
            TensorOrder::F => shape.new_f_contig(Some(layout.offset)),
        },
        (true, false) => shape.new_c_contig(Some(layout.offset)),
        (false, true) => shape.new_f_contig(Some(layout.offset)),
        (false, false) => rstsr_raise!(InvalidLayout, "Assumes contiguous layout.")?,
    };
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, new_layout)) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// Assuming contiguous array, reshapes an array without changing its data.
    ///
    /// # See also
    ///
    /// [`reshape_assume_contig`]
    pub fn reshape_assume_contig<D2>(&self, shape: D2) -> TensorBase<DataRef<'_, R::Data>, D2>
    where
        D2: DimAPI,
    {
        reshape_assume_contig(self.view(), shape)
    }

    pub fn reshape_assume_contig_f<D2>(
        &self,
        shape: D2,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
    {
        reshape_assume_contig_f(self.view(), shape)
    }

    pub fn to_shape_assume_contig<D2>(&self, shape: D2) -> TensorBase<DataRef<'_, R::Data>, D2>
    where
        D2: DimAPI,
    {
        reshape_assume_contig(self.view(), shape)
    }

    pub fn to_shape_assume_contig_f<D2>(
        &self,
        shape: D2,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
    {
        reshape_assume_contig_f(self.view(), shape)
    }

    pub fn into_shape_assume_contig<D2>(self, shape: D2) -> TensorBase<R, D2>
    where
        D2: DimAPI,
    {
        reshape_assume_contig(self, shape)
    }

    pub fn into_shape_assume_contig_f<D2>(self, shape: D2) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
    {
        reshape_assume_contig_f(self, shape)
    }
}

/* #endregion */

/* #region reshape */

/// Reshapes an array without changing its data.
///
/// # Todo
///
/// Current implementation only prohibits memory copy when the input tensor is
/// c-contiguous or f-contiguous. However, it is also possible in some other
/// cases, and we haven't implement that way.
///
/// # See also
///
/// [Python array API standard: `reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html)
pub fn reshape<'a, R, T, D, B, D2>(
    tensor: TensorBase<R, D>,
    shape: D2,
) -> TensorBase<DataCow<'a, R::Data>, D2>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
    DataCow<'a, Storage<T, B>>: From<R>,
{
    reshape_f(tensor, shape).unwrap()
}

pub fn reshape_f<'a, R, T, D, B, D2>(
    tensor: TensorBase<R, D>,
    shape: D2,
) -> Result<TensorBase<DataCow<'a, R::Data>, D2>>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    D2: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D2, D>,
    DataCow<'a, Storage<T, B>>: From<R>,
{
    rstsr_assert_eq!(tensor.size(), shape.shape_size(), InvalidLayout)?;
    let contig = tensor.layout().c_contig() || tensor.layout().f_contig();
    if contig {
        // contiguous, no data cloned
        let result = tensor.into_shape_assume_contig_f(shape.clone())?;
        let layout = result.layout().clone();
        let data = result.data.into();
        return unsafe { Ok(TensorBase::new_unchecked(data, layout)) };
    } else {
        // non-contiguous, clone data by assign
        let device = tensor.data.storage().device();
        let layout_new = shape.new_contig(None);
        let mut storage_new = unsafe { device.empty_impl(layout_new.size())? };
        device.assign_arbitary(&mut storage_new, &layout_new, tensor.storage(), tensor.layout())?;
        let data_new = DataCow::Owned(storage_new.into());
        return unsafe { Ok(TensorBase::new_unchecked(data_new, layout_new)) };
    }
}

impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Reshapes an array without changing its data.
    ///
    /// # See also
    ///
    /// [`reshape`]
    pub fn reshape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        reshape(self.view(), shape)
    }

    pub fn reshape_f<D2>(&self, shape: D2) -> Result<TensorBase<DataCow<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        reshape_f(self.view(), shape)
    }

    pub fn to_shape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        reshape(self.view(), shape)
    }

    pub fn to_shape_f<D2>(&self, shape: D2) -> Result<TensorBase<DataCow<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        reshape_f(self.view(), shape)
    }

    pub fn into_shape<'a, D2>(self, shape: D2) -> TensorBase<DataCow<'a, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
        DataCow<'a, Storage<T, B>>: From<R>,
    {
        reshape(self, shape)
    }

    pub fn into_shape_f<'a, D2>(self, shape: D2) -> Result<TensorBase<DataCow<'a, R::Data>, D2>>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
        DataCow<'a, Storage<T, B>>: From<R>,
    {
        reshape_f(self, shape)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_to_shape_assume_contig() {
        let a = linspace((2.5, 3.2, 16));
        let b = a.to_shape_assume_contig_f([4, 4]).unwrap();
        println!("{:.3?}", b);
    }

    #[test]
    fn test_expand_dims() {
        let a: Tensor<f64, _> = zeros([4, 9, 8]);
        let b = a.expand_dims(2);
        assert_eq!(b.shape(), &[4, 9, 1, 8]);
    }

    #[test]
    fn test_flip() {
        let a = arange(24.0).into_shape([2, 3, 4]).into_owned();
        println!("{:?}", a);

        let b = a.flip(1);
        println!("{:?}", b);
        assert_eq!(b.shape(), &[2, 3, 4]);
        let c = a.flip([0, -1]);
        println!("{:?}", c);
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_to_shape() {
        let a = linspace((0.0, 15.0, 16));
        let mut a = a.to_shape([4, 4]);
        a.layout = Layout::new([2, 2], [2, 4], 0);
        println!("{:?}", a);
        let b = a.to_shape([2, 2]);
        println!("{:?}", b);
    }

    #[test]
    fn test_broadcast_to() {
        let a = linspace((0.0, 15.0, 16));
        let a = a.into_shape_assume_contig_f([4, 1, 4]).unwrap();
        let a = a.to_broadcast_f([6, 4, 3, 4]).unwrap();
        assert_eq!(a.layout(), unsafe { &Layout::new_unchecked([6, 4, 3, 4], [0, 4, 0, 1], 0) });
        println!("{:?}", a);
    }
}
