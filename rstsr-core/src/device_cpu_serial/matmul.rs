//! Matrix multiplication for CPU backend.
//!
//! **This implementation is not optimized!**

use core::ops::{Add, Mul};

use crate::prelude_dev::*;

impl<TA, TB, TC, DA, DB, DC> DeviceMatMulAPI<TA, TB, TC, DA, DB, DC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
    Self: DeviceGEMMAPI<TA, TB, TC>
        + DeviceGEMVAPI<TA, TB, TC>
        + DeviceInnerDotAPI<TA, TB, TC>
        + DeviceAPI<TA, Raw = Vec<TA>>
        + DeviceAPI<TB, Raw = Vec<TB>>
        + DeviceAPI<TC, Raw = Vec<TC>>,
{
    fn matmul(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<DC>,
        a: &Vec<TA>,
        la: &Layout<DA>,
        b: &Vec<TB>,
        lb: &Layout<DB>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        match (la.ndim(), lb.ndim(), lc.ndim()) {
            (1, 1, 0) => {
                // rule 1: vector inner dot
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix0>().unwrap();
                self.inner_dot(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (2, 2, 2) => {
                // rule 2: matrix multiplication
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix2>().unwrap();
                self.gemm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (2, 1, 1) => {
                // rule 4 special: 2 x 1
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gemv(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 2, 1) => {
                // rule 3 special: 1 x 2
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let lc = &lc.clone().into_dim::<Ix1>().unwrap();
                self.gevm(c, lc, a, la, b, lb, alpha, beta)?;
            },
            (1, 2.., _) => {
                // rule 3: | `        K` | `..., K, N` | `   ..., N` |
                rstsr_assert_eq!(lb.ndim(), lc.ndim() + 1, InvalidLayout)?;
                let la = &la.clone().into_dim::<Ix1>().unwrap();
                let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-1)?;
                let lb_matmul = &mut lb_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix1>()?;
                let l_rest = translate_to_col_major(&[&lc_rest, &lb_rest], TensorIterOrder::K)?;
                let (lc_rest, lb_rest) = (&l_rest[0], &l_rest[1]);
                let itb_rest = IterLayoutColMajor::new(lb_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ib_rest, ic_rest) in izip!(itb_rest, itc_rest) {
                    unsafe { lb_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gevm(c, lc_matmul, a, la, b, lb_matmul, alpha.clone(), beta.clone())?;
                }
            },
            (2.., 1, _) => {
                // rule 4: | `..., M, K` | `        K` | `   ..., M` |
                rstsr_assert_eq!(la.ndim(), lc.ndim() + 1, InvalidLayout)?;
                let lb = &lb.clone().into_dim::<Ix1>().unwrap();
                let (la_rest, la_matmul) = la.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-1)?;
                let la_matmul = &mut la_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix1>()?;
                let l_rest = translate_to_col_major(&[&lc_rest, &la_rest], TensorIterOrder::K)?;
                let (lc_rest, la_rest) = (&l_rest[0], &l_rest[1]);
                let ita_rest = IterLayoutColMajor::new(la_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ib_rest, ic_rest) in izip!(ita_rest, itc_rest) {
                    unsafe { la_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gemv(c, lc_matmul, a, la_matmul, b, lb, alpha.clone(), beta.clone())?;
                }
            },
            (2, 3.., _) => {
                // rule 5: | `     M, K` | `..., K, N` | `..., M, N` |
                rstsr_assert_eq!(lb.ndim(), lc.ndim(), InvalidLayout)?;
                let la = &la.clone().into_dim::<Ix2>().unwrap();
                let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
                let lb_matmul = &mut lb_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix2>()?;
                let l_rest = translate_to_col_major(&[&lc_rest, &lb_rest], TensorIterOrder::K)?;
                let (lc_rest, lb_rest) = (&l_rest[0], &l_rest[1]);
                let itb_rest = IterLayoutColMajor::new(lb_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ib_rest, ic_rest) in izip!(itb_rest, itc_rest) {
                    unsafe { lb_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gemm(c, lc_matmul, a, la, b, lb_matmul, alpha.clone(), beta.clone())?;
                }
            },
            (3.., 2, _) => {
                // rule 6: | `..., M, K` | `     K, N` | `..., M, N` |
                rstsr_assert_eq!(la.ndim(), lc.ndim(), InvalidLayout)?;
                let lb = &lb.clone().into_dim::<Ix2>().unwrap();
                let (la_rest, la_matmul) = la.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
                let la_matmul = &mut la_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix2>()?;
                let l_rest = translate_to_col_major(&[&lc_rest, &la_rest], TensorIterOrder::K)?;
                let (lc_rest, la_rest) = (&l_rest[0], &l_rest[1]);
                let ita_rest = IterLayoutColMajor::new(la_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ib_rest, ic_rest) in izip!(ita_rest, itc_rest) {
                    unsafe { la_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gemm(c, lc_matmul, a, la_matmul, b, lb, alpha.clone(), beta.clone())?;
                }
            },
            (3.., 3.., _) => {
                // rule 7: | `..., M, K` | `..., K, N` | `..., M, N` |
                rstsr_assert_eq!(la.ndim(), lc.ndim(), InvalidLayout)?;
                rstsr_assert_eq!(lb.ndim(), lc.ndim(), InvalidLayout)?;
                let (la_rest, la_matmul) = la.dim_split_at(-2)?;
                let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
                let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
                let la_matmul = &mut la_matmul.into_dim::<Ix2>()?;
                let lb_matmul = &mut lb_matmul.into_dim::<Ix2>()?;
                let lc_matmul = &mut lc_matmul.into_dim::<Ix2>()?;
                let l_rest =
                    translate_to_col_major(&[&lc_rest, &la_rest, &lb_rest], TensorIterOrder::K)?;
                let (lc_rest, la_rest, lb_rest) = (&l_rest[0], &l_rest[1], &l_rest[2]);
                let ita_rest = IterLayoutColMajor::new(la_rest)?;
                let itb_rest = IterLayoutColMajor::new(lb_rest)?;
                let itc_rest = IterLayoutColMajor::new(lc_rest)?;
                for (ia_rest, ib_rest, ic_rest) in izip!(ita_rest, itb_rest, itc_rest) {
                    unsafe { la_matmul.set_offset(ia_rest) };
                    unsafe { lb_matmul.set_offset(ib_rest) };
                    unsafe { lc_matmul.set_offset(ic_rest) };
                    self.gemm(
                        c,
                        lc_matmul,
                        a,
                        la_matmul,
                        b,
                        lb_matmul,
                        alpha.clone(),
                        beta.clone(),
                    )?;
                }
            },
            // handle other cases
            (0, _, _) | (_, 0, _) // zero-dimension input
            | (1, 1, 1..) // rule 1 invalid
            | (2, 2, 3..) | (2, 2, 0..2) // rule 2 invalid
            => {
                rstsr_raise!(
                    InvalidLayout,
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

impl<TA, TB, TC> DeviceGEMMAPI<TA, TB, TC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemm(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<Ix2>,
        a: &Vec<TA>,
        la: &Layout<Ix2>,
        b: &Vec<TB>,
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
        let (m, n, k) = (sc[0], sc[1], sa[1]);

        // naive iteration: assuming c-prefer
        unsafe {
            for i_m in 0..m {
                for i_n in 0..n {
                    let idx_c = lc.index_uncheck(&[i_m, i_n]) as usize;
                    c[idx_c] = beta.clone() * c[idx_c].clone();
                }
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_m, i_k]) as usize;
                    for i_n in 0..n {
                        let idx_c = lc.index_uncheck(&[i_m, i_n]) as usize;
                        let idx_b = lb.index_uncheck(&[i_k, i_n]) as usize;
                        c[idx_c] = alpha.clone() * (a[idx_a].clone() * b[idx_b].clone())
                            + c[idx_c].clone();
                    }
                }
            }
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceGEMVAPI<TA, TB, TC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn gemv(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<Ix1>,
        a: &Vec<TA>,
        la: &Layout<Ix2>,
        b: &Vec<TB>,
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
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]) as usize;
                c[idx_c] = beta.clone() * c[idx_c].clone();
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_n, i_k]) as usize;
                    let idx_b = lb.index_uncheck(&[i_k]) as usize;
                    c[idx_c] =
                        alpha.clone() * (a[idx_a].clone() * b[idx_b].clone()) + c[idx_c].clone();
                }
            }
        }
        return Ok(());
    }

    fn gevm(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<Ix1>,
        a: &Vec<TA>,
        la: &Layout<Ix1>,
        b: &Vec<TB>,
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
        unsafe {
            for i_n in 0..n {
                let idx_c = lc.index_uncheck(&[i_n]) as usize;
                c[idx_c] = beta.clone() * c[idx_c].clone();
                for i_k in 0..k {
                    let idx_a = la.index_uncheck(&[i_k]) as usize;
                    let idx_b = lb.index_uncheck(&[i_k, i_n]) as usize;
                    c[idx_c] =
                        alpha.clone() * (a[idx_a].clone() * b[idx_b].clone()) + c[idx_c].clone();
                }
            }
        }
        return Ok(());
    }
}

impl<TA, TB, TC> DeviceInnerDotAPI<TA, TB, TC> for DeviceCpuSerial
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    fn inner_dot(
        &self,
        c: &mut Vec<TC>,
        lc: &Layout<Ix0>,
        a: &Vec<TA>,
        la: &Layout<Ix1>,
        b: &Vec<TB>,
        lb: &Layout<Ix1>,
        alpha: TC,
        beta: TC,
    ) -> Result<()> {
        // shape check
        let sa = la.shape();
        let sb = lb.shape();
        rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
        let n = sa[0];

        // naive iteration
        unsafe {
            let idx_c = lc.index_uncheck(&[]) as usize;
            let mut sum = beta * c[idx_c].clone();
            for i in 0..n {
                let idx_a = la.index_uncheck(&[i]) as usize;
                let idx_b = lb.index_uncheck(&[i]) as usize;
                sum = sum + alpha.clone() * (a[idx_a].clone() * b[idx_b].clone());
            }
            c[0] = sum;
        }
        return Ok(());
    }
}
