use crate::prelude_dev::*;
use core::mem::transmute;

/// Vector dot product of two arrays.
///
/// Let $\mathbf{a}$ be a vector in `a` and $\mathbf{b}$ be
/// a corresponding vector in `b`. The dot product is defined as:
///
/// $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i$$
///
/// where the sum is over the dimension specified by `axis` (default: last axis)
/// and where $\overline{a_i}$ denotes the complex conjugate if $a_i$
/// is complex and the identity otherwise.
///
/// # Parameters
///
/// - `a`: impl [`TensorViewAPI`]
///
///   - The first input array. Note this array is conjugated if it has a complex data type.
///   - Scalar not allowed.
///
/// - `b`: impl [`TensorViewAPI`]
///
///   - The second input array.
///   - Scalar not allowed.
///
/// - `axis`: `impl Into<Option<isize>>`
///
///   - The axis over which to compute the dot product.
///   - Default: `-1` (the last axis).
///   - If negative, the axis is counted from the last axis of each input array.
///
/// # Returns
///
/// [`Tensor<T::Output, B, DA::Max>`]
///
/// - The result shape is the broadcast of the input shapes with the contracted axis removed.
///
/// # Examples
///
/// Basic vector dot product:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([1, 2, 3], &device);
/// let b = rt::tensor_from_nested!([4, 5, 6], &device);
/// let result = rt::vecdot(&a, &b, None);
/// println!("{result}");
/// // 32
/// ```
///
/// 2-dim dot product:
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[1, 2], [3, 4]], &device);
/// let b = rt::tensor_from_nested!([[5, 6], [7, 8]], &device);
/// let result = rt::vecdot(&a, &b, None);
/// println!("{result}");
/// // [ 17 53]
/// ```
///
/// 2-dim broadcasted dot product (note in this case, the following two tensors only can be
/// broadcasted row-major):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// device.set_default_order(RowMajor);
/// let a = rt::tensor_from_nested!([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]], &device);
/// let b = rt::tensor_from_nested!([0., 0.6, 0.8], &device);
/// let result = rt::vecdot(&a, &b, None);
/// println!("{result}");
/// // [ 3 8 10]
/// ```
///
/// Complex vector dot product (conjugates first argument):
///
/// ```rust
/// # use rstsr::prelude::*;
/// # let mut device = DeviceCpu::default();
/// # device.set_default_order(RowMajor);
/// use num::complex::c64;
/// let a = rt::tensor_from_nested!([c64(1., 0.), c64(2., 2.), c64(3., 0.)], &device);
/// let b = rt::tensor_from_nested!([c64(1., 0.), c64(2., 0.), c64(3., 3.)], &device);
/// let result = rt::vecdot(&a, &b, None);
/// println!("{result}");
/// // 14+5i
/// ```
///
/// # Notes of API accordance
///
/// - Array-API: `vecdot(x1, x2, /, *, axis=-1)` ([`vecdot`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.vecdot.html))
/// - NumPy: `vecdot(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True[, signature, axes, axis])` ([`numpy.vecdot`](https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html))
/// - RSTSR: `rt::vecdot(a, b, axis)`
///
/// # Panics
///
/// - The contracted axis dimensions do not match.
/// - The input tensors cannot be broadcast together.
///
/// For a fallible version, use [`vecdot_f`].
///
/// # See Also
///
/// ## Related functions in RSTSR
///
/// - [`matmul`] - Matrix-matrix product.
/// - [`rt::tblis::tensordot`](https://docs.rs/rstsr-tblis/latest/rstsr_tblis/tensordot_impl/fn.tensordot.html)
///   - Tensor dot product along specified axes.
/// - [`rt::tblis::einsum`](https://docs.rs/rstsr-tblis/latest/rstsr_tblis/einsum_impl/fn.einsum.html)
///   - Einstein summation for tensors.
///
/// ## Variants of this function
///
/// - [`vecdot`] / [`vecdot_f`]: Returning a new tensor.
/// - [`vecdot_from`] / [`vecdot_from_f`]: Writing result to existing tensor.
pub fn vecdot<TA, TB, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl Into<Option<isize>>,
) -> Tensor<TA::Output, B, IxD>
where
    TA: Mul<TB>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TA::Output, DA, DB, IxD>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    vecdot_f(a, b, axis).rstsr_unwrap()
}

/// Vector dot product of two arrays.
///
/// See also [`vecdot`].
pub fn vecdot_from<TA, TB, TC, DA, DB, DC, B>(
    c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl Into<Option<isize>>,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TC, DA, DB, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TC>
        + DeviceAPI<MaybeUninit<TC>>,
{
    vecdot_from_f(c, a, b, axis)
}

/// Vector dot product of two arrays.
///
/// See also [`vecdot`].
pub fn vecdot_f<TA, TB, DA, DB, B>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl Into<Option<isize>>,
) -> Result<Tensor<TA::Output, B, IxD>>
where
    TA: Mul<TB>,
    DA: DimAPI,
    DB: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TA::Output, DA, DB, IxD>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
{
    let (a, b) = (a.view(), b.view());

    // check devices
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // check axis
    let axis = axis.into().unwrap_or(-1);
    let (axis_a, axis_b) = if axis < 0 {
        rstsr_pattern!(
            axis,
            -(a.ndim().min(b.ndim()) as isize)..=-1,
            InvalidValue,
            "axis should be [-N, -1] where N is min(a.ndim, b.ndim)"
        )?;
        let axis_a = (axis + a.ndim() as isize) as usize;
        let axis_b = (axis + b.ndim() as isize) as usize;
        (axis_a, axis_b)
    } else {
        rstsr_pattern!(
            axis,
            0..(a.ndim().min(b.ndim()) as isize),
            InvalidValue,
            "axis should be [0, N) where N is min(a.ndim, b.ndim)"
        )?;
        (axis as usize, axis as usize)
    };

    // chop out shape_a[axis_a] and shape_b[axis_b], and the broadcasted is c's shape
    let mut shape_a = a.shape().as_ref().to_vec();
    let mut shape_b = b.shape().as_ref().to_vec();
    let ncontract_a = shape_a.remove(axis_a);
    let ncontract_b = shape_b.remove(axis_b);
    rstsr_assert_eq!(
        ncontract_a,
        ncontract_b,
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;
    // get output layout
    let la_chop = a.layout().to_dim::<IxD>()?.dim_chop(axis_a as isize)?;
    let lb_chop = b.layout().to_dim::<IxD>()?.dim_chop(axis_b as isize)?;
    let default_order = a.device().default_order();
    let (la_chop_b, lb_chop_b) = broadcast_layout(&la_chop, &lb_chop, default_order)?;
    // generate output layout
    let layout_c = match TensorIterOrder::default() {
        TensorIterOrder::C => la_chop_b.shape().c(),
        TensorIterOrder::F => la_chop_b.shape().f(),
        _ => get_layout_for_binary_op(&la_chop_b, &lb_chop_b, default_order)?,
    };
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.vecdot(storage_c.raw_mut(), &layout_c, a.raw(), a.layout(), b.raw(), b.layout(), axis)?;
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
}

/// Vector dot product of two arrays.
///
/// See also [`vecdot`].
pub fn vecdot_from_f<TA, TB, TC, DA, DB, DC, B>(
    mut c: impl TensorViewMutAPI<Type = TC, Backend = B, Dim = DC>,
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    axis: impl Into<Option<isize>>,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TC, DA, DB, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TC>
        + DeviceAPI<MaybeUninit<TC>>,
{
    let (a, b, mut c) = (a.view(), b.view(), c.view_mut());

    // check devices
    let device = c.device().clone();
    rstsr_assert!(device.same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;

    // check axis
    let axis = axis.into().unwrap_or(-1);
    let (axis_a, axis_b) = if axis < 0 {
        rstsr_pattern!(
            axis,
            -(a.ndim().min(b.ndim()) as isize)..=-1,
            InvalidValue,
            "axis should be [-N, -1] where N is min(a.ndim, b.ndim)"
        )?;
        let axis_a = (axis + a.ndim() as isize) as usize;
        let axis_b = (axis + b.ndim() as isize) as usize;
        (axis_a, axis_b)
    } else {
        rstsr_pattern!(
            axis,
            0..(a.ndim().min(b.ndim()) as isize),
            InvalidValue,
            "axis should be [0, N) where N is min(a.ndim, b.ndim)"
        )?;
        (axis as usize, axis as usize)
    };

    // chop out shape_a[axis_a] and shape_b[axis_b], and check the rest to be broadcastable
    let mut shape_a = a.shape().as_ref().to_vec();
    let mut shape_b = b.shape().as_ref().to_vec();
    let ncontract_a = shape_a.remove(axis_a);
    let ncontract_b = shape_b.remove(axis_b);
    rstsr_assert_eq!(
        ncontract_a,
        ncontract_b,
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;
    let shape_c_expect = broadcast_shapes_f(&[shape_a, shape_b], device.default_order())?;
    let shape_c = c.shape();
    rstsr_assert_eq!(shape_c_expect, shape_c.as_ref(), InvalidLayout, "incompatible shapes in vecdot")?;

    let c_layout = c.layout().clone();
    let c_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TC>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TC>>>::Raw>(c.raw_mut())
    };
    device.vecdot(c_raw_mut, &c_layout, a.raw(), a.layout(), b.raw(), b.layout(), axis)
}

#[cfg(test)]
mod test {
    use rstsr::prelude::*;

    #[test]
    fn test_vecdot() {
        let mut device = DeviceCpuSerial::default();
        device.set_default_order(RowMajor);
        let a = rt::arange((6, &device)).into_shape((2, 3));
        let b = rt::arange((6, 12, &device)).into_shape((2, 3));
        let c = rt::vecdot(&a, &b, None);
        println!("Result c: {c}");
        let target = rt::tensor_from_nested!([23, 122], &device);
        assert!(rt::allclose(&c, &target, None));

        let a = rt::tensor_from_nested!([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]], &device);
        let b = rt::tensor_from_nested!([0., 0.6, 0.8], &device);
        let c = rt::vecdot(&a, &b, None);
        println!("Result c: {c}");
        let target = rt::tensor_from_nested!([3., 8., 10.], &device);
        assert!(rt::allclose(&c, &target, None));
    }
}
