//! Matrix multiplication for CPU backend.
//!
//! **This implementation is not optimized!**

use crate::prelude_dev::*;
use core::ops::{Add, Mul};

#[allow(clippy::too_many_arguments)]
pub fn matmul_naive_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [TC],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
    // NOTE: this only works for row-major layout
    // for column-major layout, we need to transpose the input:
    // C = A * B  =>  C^T = B^T * A^T
    match (la.ndim(), lb.ndim(), lc.ndim()) {
        (1, 1, 0) => {
            // rule 1: vector inner dot
            let la = &la.clone().into_dim::<Ix1>().unwrap();
            let lb = &lb.clone().into_dim::<Ix1>().unwrap();
            let lc = &lc.clone().into_dim::<Ix0>().unwrap();
            inner_dot_naive_cpu_serial(c, lc, a, la, b, lb, alpha, beta)?;
        },
        (2, 2, 2) => {
            // rule 2: matrix multiplication
            let la = &la.clone().into_dim::<Ix2>().unwrap();
            let lb = &lb.clone().into_dim::<Ix2>().unwrap();
            let lc = &lc.clone().into_dim::<Ix2>().unwrap();
            gemm_naive_cpu_serial(c, lc, a, la, b, lb, alpha, beta)?;
        },
        _ => {
            // broadcasted rules 3..7: the config resolves the rule, broadcasts
            // the batch (`rest`) dims against `lc`, and hands us 2-D matmul
            // layouts (via `dim_insert` for the vector rules).
            let cfg = layout_matmul_dyn_row_major_with_lc(&la.to_dim()?, &lb.to_dim()?, &lc.to_dim()?)?;
            let la_matmul = cfg.la_matmul.into_dim::<Ix2>()?;
            let lb_matmul = cfg.lb_matmul.into_dim::<Ix2>()?;
            let lc_matmul = cfg.lc_matmul.into_dim::<Ix2>()?;
            let la_rest = cfg.la_rest.unwrap();
            let lb_rest = cfg.lb_rest.unwrap();
            let lc_rest = cfg.lc_rest.unwrap();
            let ita_rest = IterLayoutColMajor::new(&la_rest)?;
            let itb_rest = IterLayoutColMajor::new(&lb_rest)?;
            let itc_rest = IterLayoutColMajor::new(&lc_rest)?;
            for (ia_rest, ib_rest, ic_rest) in izip!(ita_rest, itb_rest, itc_rest) {
                let mut la_m = la_matmul.clone();
                let mut lb_m = lb_matmul.clone();
                let mut lc_m = lc_matmul.clone();
                unsafe {
                    la_m.set_offset(ia_rest);
                    lb_m.set_offset(ib_rest);
                    lc_m.set_offset(ic_rest);
                }
                gemm_naive_cpu_serial(c, &lc_m, a, &la_m, b, &lb_m, alpha.clone(), beta.clone())?;
            }
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_naive_cpu_serial<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
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
                    c[idx_c] = alpha.clone() * (a[idx_a].clone() * b[idx_b].clone()) + c[idx_c].clone();
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gemv_naive_cpu_serial<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix1>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix1>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
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
                c[idx_c] = alpha.clone() * (a[idx_a].clone() * b[idx_b].clone()) + c[idx_c].clone();
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gevm_naive_cpu_serial<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix1>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
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
                c[idx_c] = alpha.clone() * (a[idx_a].clone() * b[idx_b].clone()) + c[idx_c].clone();
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn inner_dot_naive_cpu_serial<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix0>,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
    alpha: TC,
    beta: TC,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone,
    TA: Mul<TB, Output = TC>,
    TC: Mul<TC, Output = TC> + Add<TC, Output = TC>,
{
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
    Ok(())
}
