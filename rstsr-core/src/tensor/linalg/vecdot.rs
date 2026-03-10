use crate::prelude_dev::*;
use core::mem::transmute;

pub fn vecdot<RA, RB, TA, TB, DA, DB, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    axis: impl Into<Option<isize>>,
) -> Tensor<TA::Output, B, DA::Max>
where
    TA: Mul<TB>,
    DA: DimAPI + DimMaxAPI<DB>,
    DB: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TA::Output, DA, DB, DA::Max>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
{
    vecdot_f(a, b, axis).rstsr_unwrap()
}

pub fn vecdot_with_output<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    axis: impl Into<Option<isize>>,
) where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TC, DA, DB, DC>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TC>
        + DeviceAPI<MaybeUninit<TC>>,
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
{
    vecdot_with_output_f(c, a, b, axis).rstsr_unwrap()
}

pub fn vecdot_with_output_f<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
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
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
{
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
    rstsr_assert_eq!(
        shape_a.remove(axis_a),
        shape_b.remove(axis_b),
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;
    let shape_c_expect = broadcast_shapes(&[shape_a, shape_b], device.default_order());
    let shape_c = c.shape();
    rstsr_assert_eq!(shape_c_expect, shape_c.as_ref(), InvalidLayout, "incompatible shapes in vecdot")?;

    let c_layout = c.layout().clone();
    let c_raw_mut = unsafe {
        transmute::<&mut <B as DeviceRawAPI<TC>>::Raw, &mut <B as DeviceRawAPI<MaybeUninit<TC>>>::Raw>(c.raw_mut())
    };
    device.vecdot(c_raw_mut, &c_layout, a.raw(), a.layout(), b.raw(), b.layout(), axis)
}

pub fn vecdot_f<RA, RB, TA, TB, DA, DB, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    axis: impl Into<Option<isize>>,
) -> Result<Tensor<TA::Output, B, DA::Max>>
where
    TA: Mul<TB>,
    DA: DimAPI + DimMaxAPI<DB>,
    DB: DimAPI,
    B: DeviceVecdotAPI<TA, TB, TA::Output, DA, DB, DA::Max>
        + DeviceAPI<TA>
        + DeviceAPI<TB>
        + DeviceAPI<TA::Output>
        + DeviceCreationAnyAPI<TA::Output>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
{
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
    rstsr_assert_eq!(
        shape_a.remove(axis_a),
        shape_b.remove(axis_b),
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;
    let shape_c = broadcast_shapes(&[shape_a, shape_b], device.default_order());
    let layout_c = shape_c.new_c_contig(None).into_dim()?;
    let mut storage_c = device.uninit_impl(layout_c.bounds_index()?.1)?;
    device.vecdot(storage_c.raw_mut(), &layout_c, a.raw(), a.layout(), b.raw(), b.layout(), axis)?;
    unsafe { Tensor::new_f(B::assume_init_impl(storage_c)?, layout_c) }
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
