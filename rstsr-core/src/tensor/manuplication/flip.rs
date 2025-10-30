use crate::prelude_dev::*;

/* #region flip */

pub fn into_flip_f<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> Result<TensorBase<S, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    let (storage, mut layout) = tensor.into_raw_parts();
    let mut axes = normalize_axes_index(axes.try_into()?, layout.ndim(), false)?;
    if axes.is_empty() {
        axes = (0..layout.ndim() as isize).collect();
    }
    for axis in axes {
        layout = layout.dim_narrow(axis, slice!(None, None, -1))?;
    }
    unsafe { Ok(TensorBase::new_unchecked(storage, layout)) }
}

/// Reverses the order of elements in an array along the given axis.
///
/// # Panics
///
/// - If some index in `axis` is greater than the number of axes in the original tensor.
///
/// # See also
///
/// [Python array API standard: `flip`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.flip.html)
pub fn flip<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> TensorView<'_, T, B, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes).rstsr_unwrap()
}

pub fn flip_f<I, R, T, B, D>(tensor: &TensorAny<R, T, B, D>, axes: I) -> Result<TensorView<'_, T, B, D>>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
{
    into_flip_f(tensor.view(), axes)
}

pub fn into_flip<I, S, D>(tensor: TensorBase<S, D>, axes: I) -> TensorBase<S, D>
where
    D: DimAPI,
    I: TryInto<AxesIndex<isize>, Error = Error>,
{
    into_flip_f(tensor, axes).rstsr_unwrap()
}

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    /// Reverses the order of elements in an array along the given axis.
    ///
    /// # See also
    ///
    /// [`flip`]
    pub fn flip<I>(&self, axis: I) -> TensorView<'_, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        flip(self, axis)
    }

    pub fn flip_f<I>(&self, axis: I) -> Result<TensorView<'_, T, B, D>>
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
    pub fn into_flip<I>(self, axis: I) -> TensorAny<R, T, B, D>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip(self, axis)
    }

    pub fn into_flip_f<I>(self, axis: I) -> Result<TensorAny<R, T, B, D>>
    where
        I: TryInto<AxesIndex<isize>, Error = Error>,
    {
        into_flip_f(self, axis)
    }
}

/* #endregion */

#[cfg(test)]
mod tests {
    #[test]
    fn doc_flip() {
        use rstsr::prelude::*;

        let mut device = DeviceCpu::default();
        device.set_default_order(RowMajor);

        let a = rt::arange((8, &device)).into_shape([2, 2, 2]);
        let a_expected = rt::tensor_from_nested!([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], &device);
        assert!(rt::allclose(&a, &a_expected, None));

        let b = a.flip(0);
        let b_expected = rt::tensor_from_nested!([[[4, 5], [6, 7]], [[0, 1], [2, 3]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
        let b_sliced = a.i(slice!(None, None, -1));
        assert!(rt::allclose(&b_sliced, &b_expected, None));

        let b = a.flip(1);
        let b_expected = rt::tensor_from_nested!([[[2, 3], [0, 1]], [[6, 7], [4, 5]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        let b = a.flip([0, -1]);
        let b_expected = rt::tensor_from_nested!([[[5, 4], [7, 6]], [[1, 0], [3, 2]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));

        let b = a.flip(None);
        let b_expected = rt::tensor_from_nested!([[[7, 6], [5, 4]], [[3, 2], [1, 0]]], &device);
        assert!(rt::allclose(&b, &b_expected, None));
    }
}
