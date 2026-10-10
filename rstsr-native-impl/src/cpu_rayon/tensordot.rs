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
    order: FlagOrder,
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
        return tensordot_naive_cpu_serial(c, lc, a, la, b, lb, axes_a, axes_b, order);
    }

    let (las, lam) = la.dim_split_axes(axes_a)?;
    let (lbs, lbm) = lb.dim_split_axes(axes_b)?;
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axes should be the same"
    )?;

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
