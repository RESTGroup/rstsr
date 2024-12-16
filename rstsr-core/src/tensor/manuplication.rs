//! This module handles tensor data manipulation.

use crate::prelude_dev::*;
use core::num::TryFromIntError;

/* #region broadcast_arrays */

/// Broadcasts any number of arrays against each other.
///
/// # See also
///
/// [Python Array API standard: `broadcast_arrays`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_arrays.html)
pub fn broadcast_arrays<R>(tensors: Vec<TensorBase<R, IxD>>) -> Result<Vec<TensorBase<R, IxD>>>
where
    R: DataAPI,
    IxD: DimMaxAPI<IxD>,
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
        let tensor = broadcast_to(tensor, &shape_b)?;
        tensors_new.push(tensor);
    }
    return Ok(tensors_new);
}

/* #endregion */

/* #region broadcast_to */

/// Broadcasts an array to a specified shape.
///
/// # See also
///
/// [Python Array API standard: `broadcast_to`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_to.html)
pub fn broadcast_to<R, D, D2>(tensor: TensorBase<R, D>, shape: &D2) -> Result<TensorBase<R, D2>>
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
    pub fn broadcast_to<D2>(&self, shape: &D2) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        broadcast_to(self.view(), shape)
    }

    /// Broadcasts an array to a specified shape.
    ///
    /// # See also
    ///
    /// [`broadcast_to`]
    pub fn into_broadcast_to<D2>(self, shape: &D2) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        D: DimMaxAPI<D2, Max = D2>,
    {
        broadcast_to(self, shape)
    }
}

/* #endregion */

/* #region expand_dims */

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
pub fn expand_dims<I, R, D>(tensor: TensorBase<R, D>, axis: I) -> TensorBase<R, D::LargerOne>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize, Error = TryFromIntError>,
{
    let axis = axis.try_into().unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_insert(axis).unwrap();
    let layout = layout.try_into().unwrap(); // safe to unwrap
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
}

/// Methods for tensor manipulation (dimension expanded).
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimLargerOneAPI,
    D::LargerOne: DimAPI,
    Layout<D::LargerOne>: TryFrom<Layout<IxD>, Error = Error>,
{
    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn expand_dims<I>(&self, axis: I) -> TensorBase<DataRef<'_, R::Data>, D::LargerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        expand_dims(self.view(), axis)
    }

    /// Expands the shape of an array by inserting a new axis (dimension) of
    /// size one at the position specified by `axis`.
    ///
    /// # See also
    ///
    /// [`expand_dims`]
    pub fn into_expand_dims<I>(self, axis: I) -> TensorBase<R, D::LargerOne>
    where
        I: TryInto<isize, Error = TryFromIntError>,
    {
        expand_dims(self, axis)
    }
}

/* #endregion */

/* #region flip */

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
pub fn flip<I, R, D>(tensor: TensorBase<R, D>, axes: &[I]) -> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize> + Copy,
{
    let mut layout = tensor.layout().clone();
    for &axis in axes.iter() {
        let axis = axis.try_into().map_err(|_| "Into isize failed").unwrap();
        layout = layout.dim_narrow(axis, slice!(None, None, -1)).unwrap();
    }
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
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
        I: TryInto<isize> + Copy,
    {
        flip(self.view(), &[axis])
    }

    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn into_flip<I>(self, axis: I) -> TensorBase<R, D>
    where
        I: TryInto<isize> + Copy,
    {
        flip(self, &[axis])
    }
}

/* #endregion */

/* #region permute_dims */

/// Permutes the axes (dimensions) of an array `x`.
///
/// # See also
///
/// - [Python array API standard: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
pub fn transpose<I, R, D>(tensor: TensorBase<R, D>, axes: &[I]) -> Result<TensorBase<R, D>>
where
    R: DataAPI,
    D: DimAPI,
    I: TryInto<isize> + Copy,
{
    let axes =
        axes.iter().map(|&x| x.try_into().map_err(|_| "Into isize failed").unwrap()).collect_vec();
    let layout = tensor.layout().transpose(&axes)?;
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

pub use transpose as permute_dims;

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
    pub fn transpose<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize> + Copy,
    {
        transpose(self.view(), axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_transpose<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize> + Copy,
    {
        transpose(self, axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn permute_dims<I>(&self, axes: &[I]) -> Result<TensorBase<DataRef<'_, R::Data>, D>>
    where
        I: TryInto<isize> + Copy,
    {
        transpose(self.view(), axes)
    }

    /// Permutes the axes (dimensions) of an array `x`.
    ///
    /// # See also
    ///
    /// [`transpose`]
    pub fn into_permute_dims<I>(self, axes: &[I]) -> Result<TensorBase<R, D>>
    where
        I: TryInto<isize> + Copy,
    {
        transpose(self, axes)
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
    I: TryInto<isize>,
{
    let axis1 = axis1.try_into().map_err(|_| "Into isize failed").unwrap();
    let axis2 = axis2.try_into().map_err(|_| "Into isize failed").unwrap();
    let layout = tensor.layout().swapaxes(axis1, axis2).unwrap();
    unsafe { TensorBase::new_unchecked(tensor.data, layout) }
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
        I: TryInto<isize>,
    {
        swapaxes(self.view(), axis1, axis2)
    }

    /// Interchange two axes of an array.
    ///
    /// # See also
    ///
    /// [`swapaxes`]
    pub fn into_swapaxes<I>(self, axis1: I, axis2: I) -> TensorBase<R, D>
    where
        I: TryInto<isize>,
    {
        swapaxes(self, axis1, axis2)
    }
}

/* #endregion */

/* #region squeeze */

/// Removes singleton dimensions (axes) from `x`.
///
/// # See also
///
/// [Python array API standard: `squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html)
pub fn squeeze<I, R, D>(tensor: TensorBase<R, D>, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
    I: TryInto<isize>,
{
    let axis = axis.try_into().map_err(|_| "Into isize failed").unwrap(); // almost safe to unwrap
    let layout = tensor.layout().dim_eliminate(axis)?;
    let layout = layout.try_into().unwrap(); // safe to unwrap
    unsafe { Ok(TensorBase::new_unchecked(tensor.data, layout)) }
}

impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    Layout<D::SmallerOne>: TryFrom<Layout<IxD>, Error = Error>,
{
    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn squeeze<I>(&self, axis: I) -> Result<TensorBase<DataRef<'_, R::Data>, D::SmallerOne>>
    where
        I: TryInto<isize>,
    {
        squeeze(self.view(), axis)
    }

    /// Removes singleton dimensions (axes) from `x`.
    ///
    /// # See also
    ///
    /// [`squeeze`]
    pub fn into_squeeze<I>(self, axis: I) -> Result<TensorBase<R, D::SmallerOne>>
    where
        I: TryInto<isize>,
    {
        squeeze(self, axis)
    }
}

/* #endregion */

/// Methods for tensor shape change without data clone.
impl<R, D> TensorBase<R, D>
where
    R: DataAPI,
    D: DimAPI,
{
    /// View of reshaped tensor.
    ///
    /// This function will raise error when
    /// - The number of elements in the reshaped tensor is not same as the
    ///   original tensor.
    /// - The layout of the original tensor is not contiguous.
    pub fn to_shape_assume_contig<D2>(
        &self,
        shape: D2,
    ) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
    {
        self.view().into_shape_assume_contig(shape)
    }

    /// View of reshaped tensor.
    ///
    /// # See also
    ///
    /// [`Tensor::to_shape_assume_contig`]
    pub fn into_shape_assume_contig<D2>(self, shape: D2) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
    {
        let layout = self.layout();
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
        unsafe { Ok(TensorBase::new_unchecked(self.data, new_layout)) }
    }

    /// Convert layout to another dimension.
    ///
    /// This is mostly used when converting static dimension to dynamic
    /// dimension or vice versa.
    pub fn to_dim<D2>(&self) -> Result<TensorBase<DataRef<'_, R::Data>, D2>>
    where
        D2: DimAPI,
        D: DimConvertAPI<D2>,
    {
        self.view().into_dim()
    }

    /// Convert layout to another dimension.
    ///
    /// # See also
    ///
    /// [`Tensor::to_dim`]
    pub fn into_dim<D2>(self) -> Result<TensorBase<R, D2>>
    where
        D2: DimAPI,
        D: DimConvertAPI<D2>,
    {
        let layout = self.layout().clone().into_dim::<D2>()?;
        unsafe { Ok(TensorBase::new_unchecked(self.data, layout)) }
    }
}

/// Methods for tensor shape change with possible data clone.
impl<R, T, D, B> TensorBase<R, D>
where
    R: DataAPI<Data = Storage<T, B>>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T>,
{
    /// Reshapes an array.
    ///
    /// Values of data may not be changed, but explicit copy of data may occur
    /// depending on whether layout is c/f-contiguous.
    ///
    /// # See also
    ///
    /// - [Python array API standard: `reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html)
    pub fn reshape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        self.to_shape(shape)
    }

    /// Reshapes an array.
    ///
    /// # See also
    ///
    /// [`Tensor::reshape`]
    pub fn to_shape<D2>(&self, shape: D2) -> TensorBase<DataCow<'_, R::Data>, D2>
    where
        D2: DimAPI,
        B: OpAssignArbitaryAPI<T, D2, D>,
    {
        rstsr_assert_eq!(self.size(), shape.shape_size(), InvalidLayout).unwrap();
        let result = self.to_shape_assume_contig(shape.clone());
        if let Ok(result) = result {
            // contiguous, no data cloned
            let layout = result.layout().clone();
            let data = result.data.into();
            return TensorBase { data, layout };
        } else {
            // non-contiguous, clone data if necessary
            let device = self.data.storage().device();
            let layout_new = shape.new_contig(None);
            let mut storage_new = unsafe { device.empty_impl(layout_new.size()).unwrap() };
            device
                .assign_arbitary(&mut storage_new, &layout_new, self.storage(), self.layout())
                .unwrap();
            let data_new = DataCow::Owned(storage_new.into());
            TensorBase { data: data_new, layout: layout_new }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::Tensor;

    #[test]
    fn test_to_shape_assume_contig() {
        let a = Tensor::linspace_cpu(2.5, 3.2, 16);
        let b = a.to_shape_assume_contig([4, 4]).unwrap();
        println!("{:.3?}", b);
    }

    // fn test_expand_dims() {
    //     let a = Tensor::<f64, _>::zeros([4, 9, 8]);
    //     let b = a.expand_dims(2);
    //     assert_eq!(b.shape(), &[4, 9, 1, 8]);
    // }
    #[test]
    fn test_flip() {
        let device = DeviceCpu::default();
        let a = Tensor::<f64, _>::new(
            Storage::<f64, DeviceCpu>::new(
                (0..24).map(|v| v as f64).collect::<Vec<_>>(),
                device.clone(),
            )
            .into(),
            [2, 3, 4].c(),
        )
        .unwrap();
        println!("{:?}", a);

        let b = a.flip(1);
        println!("{:?}", b);
        assert_eq!(b.shape(), &[2, 3, 4]);
        let c = a.flip(2);
        println!("{:?}", c);
        assert_eq!(c.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_to_shape() {
        let a = Tensor::<f64, _>::linspace_cpu(0.0, 15.0, 16);
        let mut a = a.to_shape([4, 4]);
        a.layout = Layout::new([2, 2], [2, 4], 0);
        println!("{:?}", a);
        let b = a.to_shape([2, 2]);
        println!("{:?}", b);
    }

    #[test]
    fn test_broadcast_to() {
        let a = Tensor::linspace_cpu(0.0, 15.0, 16);
        let a = a.into_shape_assume_contig([4, 1, 4]).unwrap();
        let a = a.broadcast_to(&[6, 4, 3, 4]).unwrap();
        assert_eq!(a.layout(), unsafe { &Layout::new_unchecked([6, 4, 3, 4], [0, 4, 0, 1], 0) });
        println!("{:?}", a);
    }
}
