//! This module is to implement parallel matrix multiplication, but by some
//! naive way.
//!
//! This implementation should not be efficient. It is just for non-blas
//! compatible types, or as reference implementation.

use crate::prelude_dev::*;
use core::ops::{Add, Mul};
use num::Zero;
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub fn gemm_naive_rayon<TA, TB, TC>(
    c: &mut [TC],
    lc: &Layout<Ix2>,
    a: &[TA],
    la: &Layout<Ix2>,
    b: &[TB],
    lb: &Layout<Ix2>,
    alpha: TC,
    beta: TC,
    pool: &ThreadPool,
) -> Result<()>
where
    TA: Clone + Send + Sync + Mul<TB, Output = TC>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;
    let (m, n, k) = (sc[0], sc[1], sa[1]);

    pool.install(|| {
        (0..n).into_par_iter().for_each(|j| {
            (0..m).into_par_iter().for_each(|i| unsafe {
                let ptr_c = c.as_ptr().offset(lc.index_uncheck(&[i, j])) as *mut TC;
                *ptr_c = (*ptr_c).clone() * beta.clone()
                    + (0..k).fold(TC::zero(), |acc, p| {
                        let val_a = a[la.index_uncheck(&[i, p]) as usize].clone();
                        let val_b = b[lb.index_uncheck(&[p, j]) as usize].clone();
                        acc + val_a * val_b
                    }) * alpha.clone();
            });
        });
    });
    return Ok(());
}

#[allow(clippy::too_many_arguments)]
pub fn inner_dot_naive_rayon<TA, TB, TC>(
    c: &mut TC,
    a: &[TA],
    la: &Layout<Ix1>,
    b: &[TB],
    lb: &Layout<Ix1>,
    alpha: TC,
    beta: TC,
    pool: &ThreadPool,
) -> Result<()>
where
    TA: Clone + Send + Sync + Mul<TB, Output = TC>,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Mul<TC, Output = TC> + Add<TC, Output = TC> + Zero,
{
    // shape check
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sa[0], sb[0], InvalidLayout)?;
    let n = sa[0];

    let c_innerdot = pool.install(|| {
        (0..n)
            .into_par_iter()
            .fold(
                || TC::zero(),
                |acc, i| unsafe {
                    acc + a[la.index_uncheck(&[i]) as usize].clone()
                        * b[lb.index_uncheck(&[i]) as usize].clone()
                },
            )
            .reduce_with(|a, b| a + b)
            .unwrap_or(TC::zero())
    });
    *c = c_innerdot * alpha + c.clone() * beta;
    return Ok(());
}
