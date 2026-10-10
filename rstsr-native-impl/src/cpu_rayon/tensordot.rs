//! Rayon twin of the naive `tensordot` kernel.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;

const PARALLEL_SWITCH: usize = 512;

/// Rayon-parallel naive tensor contraction; falls back to the serial kernel
/// below `PARALLEL_SWITCH` or without a pool. Parallel over output elements,
/// sequential reduction over contracted ones.
pub fn tensordot_naive_cpu_rayon<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    axes_a: &[isize],
    axes_b: &[isize],
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Clone + Send + Sync,
    TB: Clone + Send + Sync,
    TC: Clone + Send + Sync + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
{
    if la.size().max(lb.size()) < PARALLEL_SWITCH || pool.is_none() {
        return tensordot_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b);
    }

    let (las, lam, lbs, lbm) = split_tensordot_axes(la, axes_a, lb, axes_b)?;

    let lc = lc.to_dim::<IxD>()?;
    let offset_a = la.offset();
    let offset_b = lb.offset();

    let shape_c = lc.shape().clone();
    let stride_a = lam.stride().iter().copied().chain(core::iter::repeat(0isize).take(lbm.ndim())).collect_vec();
    let stride_b = core::iter::repeat(0isize).take(lam.ndim()).chain(lbm.stride().iter().copied()).collect_vec();
    // SAFETY: both are valid (read-only) broadcast views of real layouts.
    let lam_e = unsafe { Layout::<IxD>::new_unchecked(shape_c.clone(), stride_a, offset_a) };
    let lbm_e = unsafe { Layout::<IxD>::new_unchecked(shape_c, stride_b, offset_b) };

    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        layout_col_major_dim_dispatch_par_3(&lc, &lam_e, &lbm_e, |(idx_c, idx_ma, idx_mb)| {
            let mut val_c: TC = Zero::zero();
            let stat = layout_col_major_dim_dispatch_2(&las, &lbs, |(idx_sa, idx_sb)| {
                let ia = idx_ma + idx_sa - offset_a;
                let ib = idx_mb + idx_sb - offset_b;
                val_c = val_c.clone() + a[ia].clone() * b[ib].clone();
            });
            stat.rstsr_unwrap();
            unsafe {
                // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`;
                // each task writes the disjoint output position `idx_c`.
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                (*c_ptr).write(val_c);
            }
        })
    };
    pool.map_or_else(task, |pool| pool.install(task))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parallel branch (operand size past `PARALLEL_SWITCH`) must agree with
    /// the serial kernel element-for-element.
    #[test]
    fn test_parallel_branch_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let (m, k, n) = (24usize, 24usize, 24usize);
        let la = Layout::<Ix2>::new([m, k], [k as isize, 1], 0).unwrap();
        let lb = Layout::<Ix2>::new([k, n], [n as isize, 1], 0).unwrap();
        let lc = Layout::<Ix2>::new([m, n], [n as isize, 1], 0).unwrap();
        let a: Vec<f64> = (0..m * k).map(|i| (i % 7) as f64 - 3.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 5) as f64 + 1.0).collect();

        let mut c_par = vec![MaybeUninit::<f64>::uninit(); m * n];
        let mut c_ser = vec![MaybeUninit::<f64>::uninit(); m * n];
        tensordot_naive_cpu_rayon(&mut c_par, &lc, &a, &la, &b, &lb, &[1], &[0], Some(&pool)).unwrap();
        tensordot_naive_cpu_serial(&mut c_ser, &lc, &a, &la, &b, &lb, &[1], &[0]).unwrap();

        let c_par: Vec<f64> = c_par.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        let c_ser: Vec<f64> = c_ser.into_iter().map(|x| unsafe { x.assume_init() }).collect();
        assert_eq!(c_par, c_ser);
    }
}
