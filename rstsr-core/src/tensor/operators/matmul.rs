//! Matrix-multiplication for tensor.

use crate::prelude_dev::*;
use core::ops::{Add, Mul, Rem};
use num::{One, Zero};

/* #region matmul by function */

pub fn op_mutc_refa_refb_matmul<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // storage
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
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
    let sa = a.raw();
    let sb = b.raw();
    let sc = c.raw_mut();
    device.matmul(sc, &lc, sa, la, sb, lb, alpha, beta)
}

pub fn op_refa_refb_matmul<RA, RB, TA, TB, TC, DA, DB, DC, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    alpha: TC,
) -> Result<Tensor<TC, B, DC>>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    rstsr_assert!(b.device().same_device(b.device()), DeviceMismatch)?;
    let default_order = a.device().default_order();
    let cfg = LayoutMatMulConfig::<DA, DB>::layout_matmul(a.layout(), b.layout(), default_order)?;
    let lc = cfg.lc;
    let mut c: Tensor<TC, B, _> = unsafe { empty((lc, a.device())) }.into_dim_f()?;
    op_mutc_refa_refb_matmul(&mut c, a, b, alpha, TC::zero())?;
    return Ok(c);
}

pub fn matmul_with_output_f<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    c: &mut TensorAny<RC, TC, B, DC>,
) -> Result<()>
where
    // storage
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, TC::one(), TC::zero())
}

pub fn matmul_with_output<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    c: &mut TensorAny<RC, TC, B, DC>,
) where
    // storage
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, TC::one(), TC::zero()).unwrap()
}

pub fn matmul_from_f<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    // storage
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, alpha, beta)
}

pub fn matmul_from<RA, RB, RC, TA, TB, TC, DA, DB, DC, B>(
    c: &mut TensorAny<RC, TC, B, DC>,
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
    alpha: TC,
    beta: TC,
) where
    // storage
    RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_mutc_refa_refb_matmul(c, a, b, alpha, beta).unwrap()
}

pub fn matmul_f<RA, RB, TA, TB, TC, DA, DB, DC, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
) -> Result<Tensor<TC, B, DC>>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_matmul(a, b, TC::one())
}

pub fn matmul<RA, RB, TA, TB, TC, DA, DB, DC, B>(
    a: &TensorAny<RA, TA, B, DA>,
    b: &TensorAny<RB, TB, B, DB>,
) -> Tensor<TC, B, DC>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    op_refa_refb_matmul(a, b, TC::one()).unwrap()
}

/* #endregion */

/* #region matmul implementation to core ops */

impl<RA, RB, TA, TB, TC, DA, DB, DC, B> Rem<&TensorAny<RB, TB, B, DB>> for &TensorAny<RA, TA, B, DA>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    type Output = Tensor<TC, B, DC>;
    fn rem(self, rhs: &TensorAny<RB, TB, B, DB>) -> Self::Output {
        op_refa_refb_matmul(self, rhs, TC::one()).unwrap()
    }
}

impl<RA, RB, TA, TB, TC, DA, DB, DC, B> Rem<&TensorAny<RB, TB, B, DB>> for TensorAny<RA, TA, B, DA>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    type Output = Tensor<TC, B, DC>;
    fn rem(self, rhs: &TensorAny<RB, TB, B, DB>) -> Self::Output {
        op_refa_refb_matmul(&self, rhs, TC::one()).unwrap()
    }
}

impl<RA, RB, TA, TB, TC, DA, DB, DC, B> Rem<TensorAny<RB, TB, B, DB>> for &TensorAny<RA, TA, B, DA>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    type Output = Tensor<TC, B, DC>;
    fn rem(self, rhs: TensorAny<RB, TB, B, DB>) -> Self::Output {
        op_refa_refb_matmul(self, &rhs, TC::one()).unwrap()
    }
}

impl<RA, RB, TA, TB, TC, DA, DB, DC, B> Rem<TensorAny<RB, TB, B, DB>> for TensorAny<RA, TA, B, DA>
where
    // storage
    RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
    RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
    // dimension
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    // operation specific
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
    B: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<TC>,
    B: DeviceCreationAnyAPI<TC>,
    LayoutMatMulConfig<DA, DB>: LayoutMatMulAPI<DA, DB, DC = DC>,
    B: DeviceMatMulAPI<TA, TB, TC, DA, DB, DC>,
{
    type Output = Tensor<TC, B, DC>;
    fn rem(self, rhs: TensorAny<RB, TB, B, DB>) -> Self::Output {
        op_refa_refb_matmul(&self, &rhs, TC::one()).unwrap()
    }
}

/* #endregion */

/* #region matmul tensor trait */

// this trait is to make a.matmul(b) be possible
// however, a.rem(b) works the same to a.matmul(b), not rem(a, b)

impl<R, T, B, D> TensorAny<R, T, B, D>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    B: DeviceAPI<T>,
    D: DimAPI,
{
    pub fn matmul_f<RB, TB, TC, DB, DC>(
        &self,
        rhs: &TensorAny<RB, TB, B, DB>,
    ) -> Result<Tensor<TC, B, DC>>
    where
        // storage
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
        B: DeviceAPI<TB> + DeviceAPI<TC>,
        B: DeviceCreationAnyAPI<TC>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_matmul(self, rhs, TC::one())
    }

    pub fn matmul<RB, TB, TC, DB, DC>(&self, rhs: &TensorAny<RB, TB, B, DB>) -> Tensor<TC, B, DC>
    where
        // storage
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
        B: DeviceAPI<TB> + DeviceAPI<TC>,
        B: DeviceCreationAnyAPI<TC>,
        LayoutMatMulConfig<D, DB>: LayoutMatMulAPI<D, DB, DC = DC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_refa_refb_matmul(self, rhs, TC::one()).unwrap()
    }

    pub fn matmul_with_output_f<RB, RC, TB, TC, DB, DC>(
        &self,
        rhs: &TensorAny<RB, TB, B, DB>,
        c: &mut TensorAny<RC, TC, B, DC>,
    ) -> Result<()>
    where
        // storage
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
        B: DeviceAPI<TB> + DeviceAPI<TC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_mutc_refa_refb_matmul(c, self, rhs, TC::one(), TC::zero())
    }

    pub fn matmul_with_output<RB, RC, TB, TC, DB, DC>(
        &self,
        rhs: &TensorAny<RB, TB, B, DB>,
        c: &mut TensorAny<RC, TC, B, DC>,
    ) where
        // storage
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        RC: DataMutAPI<Data = <B as DeviceRawAPI<TC>>::Raw>,
        // dimension
        DB: DimAPI,
        DC: DimAPI,
        // operation specific
        T: Mul<TB, Output = TC>,
        TC: Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero + One,
        B: DeviceAPI<TB> + DeviceAPI<TC>,
        B: DeviceMatMulAPI<T, TB, TC, D, DB, DC>,
    {
        op_mutc_refa_refb_matmul(c, self, rhs, TC::one(), TC::zero()).unwrap()
    }

    pub fn matmul_from_f<RA, RB, TA, TB, DA, DB>(
        &mut self,
        a: &TensorAny<RA, TA, B, DA>,
        b: &TensorAny<RB, TB, B, DB>,
        alpha: T,
        beta: T,
    ) -> Result<()>
    where
        // storage
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // dimension
        DA: DimAPI,
        DB: DimAPI,
        // operation specific
        TA: Mul<TB, Output = T>,
        T: Mul<T, Output = T> + Add<T, Output = T> + Zero + One,
        B: DeviceAPI<TA> + DeviceAPI<TB>,
        B: DeviceMatMulAPI<TA, TB, T, DA, DB, D>,
    {
        op_mutc_refa_refb_matmul(self, a, b, alpha, beta)
    }

    pub fn matmul_from<RA, RB, TA, TB, DA, DB>(
        &mut self,
        a: &TensorAny<RA, TA, B, DA>,
        b: &TensorAny<RB, TB, B, DB>,
        alpha: T,
        beta: T,
    ) where
        // storage
        R: DataMutAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
        RA: DataAPI<Data = <B as DeviceRawAPI<TA>>::Raw>,
        RB: DataAPI<Data = <B as DeviceRawAPI<TB>>::Raw>,
        // dimension
        DA: DimAPI,
        DB: DimAPI,
        // operation specific
        TA: Mul<TB, Output = T>,
        T: Mul<T, Output = T> + Add<T, Output = T> + Zero + One,
        B: DeviceAPI<TA> + DeviceAPI<TB>,
        B: DeviceMatMulAPI<TA, TB, T, DA, DB, D>,
    {
        op_mutc_refa_refb_matmul(self, a, b, alpha, beta).unwrap()
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = linspace((0.0, 14.0, 15)).into_shape_assume_contig([3, 5]);
        let b = linspace((0.0, 14.0, 15)).into_shape_assume_contig([5, 3]);
        let mut c: Tensor<f64> = zeros([3, 3]);

        op_mutc_refa_refb_matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();
        println!("{c}");

        let d = &a % &b;
        println!("{d}");

        let a = linspace((0.0, 14.0, 15));
        let b = linspace((0.0, 14.0, 15));
        println!("{:}", &a % &b);

        let a = linspace((0.0, 2.0, 3));
        let b = linspace((0.0, 29.0, 30)).into_shape_assume_contig([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = linspace((0.0, 29.0, 30)).into_shape_assume_contig([2, 3, 5]);
        let b = linspace((0.0, 4.0, 5));
        println!("{:}", &a % &b);

        let a = linspace((0.0, 14.0, 15)).into_shape_assume_contig([5, 3]);
        let b = linspace((0.0, 29.0, 30)).into_shape_assume_contig([2, 3, 5]);
        println!("{:}", &a % &b);

        let a = linspace((0.0, 29.0, 30)).into_shape_assume_contig([2, 3, 5]);
        let b = linspace((0.0, 14.0, 15)).into_shape_assume_contig([5, 3]);
        println!("{:}", &a % &b);
    }

    #[test]
    fn test_matmul_from() {
        let a = linspace((0.0, 14.0, 15)).into_shape([3, 5]);
        let b = linspace((0.0, 19.0, 20)).into_shape([5, 4]);
        let mut c = linspace((0.0, 11.0, 12)).into_shape([3, 4]);
        c.matmul_from(&a, &b, 2.0, 1.5);
        println!("{c}");

        let c_ref_vec =
            vec![240., 261.5, 283., 304.5, 646., 717.5, 789., 860.5, 1052., 1173.5, 1295., 1416.5];
        let c_ref = asarray(c_ref_vec);
        assert!(allclose_f64(&c, &c_ref));
    }
}
