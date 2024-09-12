//! Matrix-multiplication for tensor.

use core::ops::{Add, Mul, Rem};

use num::{One, Zero};

use crate::prelude_dev::*;

pub fn op_mutc_refa_refb_matmul<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorBase<RC, DC>,
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // storage
    RC: DataMutAPI<Data = Storage<TC, B>>,
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    rstsr_assert!(c.device().same_device(a.device()), DeviceMismatch)?;
    rstsr_assert!(c.device().same_device(b.device()), DeviceMismatch)?;
    let device = c.device().clone();
    let la = a.layout();
    let lb = b.layout();
    let lc = c.layout().clone();
    let sa = a.data().storage();
    let sb = b.data().storage();
    let sc = c.data_mut().storage_mut();
    device.matmul(sc, &lc, sa, la, sb, lb, alpha, beta)
}

#[allow(clippy::type_complexity)]
pub fn op_refa_refb_matmul<RA, RB, TA, TB, TC, DA, DB, B>(
    a: &TensorBase<RA, DA>,
    b: &TensorBase<RB, DB>,
    alpha: TC,
) -> Result<Tensor<TC, <LayoutMatMulConfig<DA, DB> as LayoutMatMulAPI<DA, DB>>::DC, B>>
where
    // storage
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB>,
    B: DeviceMatMulAPI<
        TA,
        TB,
        TC,
        DA,
        DB,
        <LayoutMatMulConfig<DA, DB> as LayoutMatMulAPI<DA, DB>>::DC,
    >,
{
    rstsr_assert!(b.device().same_device(b.device()), DeviceMismatch)?;
    let cfg = LayoutMatMulConfig::<DA, DB>::layout_matmul(
        a.layout(),
        b.layout(),
        TensorIterOrder::default(),
    )?;
    let lc = cfg.lc;
    let mut c = unsafe { Tensor::<TC, _, B>::empty(lc, a.device()) };
    op_mutc_refa_refb_matmul(&mut c, a, b, alpha, TC::zero())?;
    return Ok(c);
}

impl<RA, RB, TA, TB, TC, DA, DB, B> Rem<&TensorBase<RB, DB>> for &TensorBase<RA, DA>
where
    // storage
    RA: DataAPI<Data = Storage<TA, B>>,
    RB: DataAPI<Data = Storage<TB, B>>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB>,
    B: DeviceMatMulAPI<
        TA,
        TB,
        TC,
        DA,
        DB,
        <LayoutMatMulConfig<DA, DB> as LayoutMatMulAPI<DA, DB>>::DC,
    >,
{
    type Output = Tensor<TC, <LayoutMatMulConfig<DA, DB> as LayoutMatMulAPI<DA, DB>>::DC, B>;
    fn rem(self, rhs: &TensorBase<RB, DB>) -> Self::Output {
        op_refa_refb_matmul(self, rhs, TC::one()).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::linspace_cpu(0.0, 14.0, 15).into_shape_assume_contig([3, 5]).unwrap();
        let b = Tensor::linspace_cpu(0.0, 14.0, 15).into_shape_assume_contig([5, 3]).unwrap();
        let mut c = Tensor::<f64, Ix2>::zeros_cpu([3, 3]);

        op_mutc_refa_refb_matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();
        println!("{c}");

        let d = &a % &b;
        println!("{d}");

        let a = Tensor::linspace_cpu(0.0, 14.0, 15);
        let b = Tensor::linspace_cpu(0.0, 14.0, 15);
        println!("{:}", &a % &b);

        let a = Tensor::linspace_cpu(0.0, 2.0, 3);
        let b = Tensor::linspace_cpu(0.0, 29.0, 30).into_shape_assume_contig([2, 3, 5]).unwrap();
        println!("{:}", &a % &b);

        let a = Tensor::linspace_cpu(0.0, 29.0, 30).into_shape_assume_contig([2, 3, 5]).unwrap();
        let b = Tensor::linspace_cpu(0.0, 4.0, 5);
        println!("{:}", &a % &b);

        let a = Tensor::linspace_cpu(0.0, 14.0, 15).into_shape_assume_contig([5, 3]).unwrap();
        let b = Tensor::linspace_cpu(0.0, 29.0, 30).into_shape_assume_contig([2, 3, 5]).unwrap();
        println!("{:}", &a % &b);

        let a = Tensor::linspace_cpu(0.0, 29.0, 30).into_shape_assume_contig([2, 3, 5]).unwrap();
        let b = Tensor::linspace_cpu(0.0, 14.0, 15).into_shape_assume_contig([5, 3]).unwrap();
        println!("{:}", &a % &b);
    }
}
