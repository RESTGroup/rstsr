//! Matrix multiplication for CPU backend.
//!
//! **This implementation is not optimized!**

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceGEMMAPI<TA, TB, TC>,
    Self: DeviceGEMVAPI<TA, TB, TC>,
    Self: DeviceInnerDotAPI<TA, TB, TC>,
{
    fn matmul(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<DC>,
        a: &Storage<TA, Self>,
        la: &Layout<DA>,
        b: &Storage<TB, Self>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        match (la.ndim(), lb.ndim(), lc.ndim()) {
            (2, 2, 2) => {
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix2>().unwrap();
                self.gemm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (2, 1, 1) => {
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gemv(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 2, 1) => {
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gevm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            _ => {
                rstsr_raise!(
                    UnImplemented,
                    "Invalid ndim for matmul: {}, {}, {}",
                    la.ndim(),
                    lb.ndim(),
                    lc.ndim()
                )?;
            },
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceGEMMAPI<TA, TB, TC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix2>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
        rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
        rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;
        let (n, m, k) = (sc[0], sc[1], sa[1]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..n {
                for i_k in 0..k {
                    let idx_b = lb.index_uncheck(&[i_k, i_n]);
                    for i_m in 0..m {
                        let idx_c = lc.index_uncheck(&[i_m, i_n]);
                        let idx_a = la.index_uncheck(&[i_m, i_k]);
                        vc[idx_c] = alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone())
                            + beta.clone() * vc[idx_c].clone();
                    }
                }
            }
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceGEMVAPI<TA, TB, TC> for CpuDevice
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemv(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix2>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
        rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
        let (n, k) = (sa[0], sa[1]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]);
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_n, i_k]);
                    let idx_b = lb.index_uncheck(&[i_k]);
                    vc[idx_c] = alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone())
                        + beta.clone() * vc[idx_c].clone();
                }
            }
        }
        return Ok(());
    }

    fn gevm(
        &self,
        c: &mut Storage<TC, Self>,
        lc: &Layout<Ix1>,
        a: &Storage<TA, Self>,
        la: &Layout<Ix1>,
        b: &Storage<TB, Self>,
        lb: &Layout<Ix2>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sc = lc.shape();
        let sa = la.shape();
        let sb = lb.shape();
        rstsr_assert_eq!(sc[0], sb[1], InvalidLayout)?;
        rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
        let (n, k) = (sb[1], sb[0]);

        // naive iteration: assuming c-prefer
        let vc = c.rawvec_mut();
        let va = a.rawvec();
        let vb = b.rawvec();
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]);
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_k]);
                    let idx_b = lb.index_uncheck(&[i_k, i_n]);
                    vc[idx_c] = alpha.clone() * (va[idx_a].clone() * vb[idx_b].clone())
                        + beta.clone() * vc[idx_c].clone();
                }
            }
        }
        return Ok(());
    }
}
