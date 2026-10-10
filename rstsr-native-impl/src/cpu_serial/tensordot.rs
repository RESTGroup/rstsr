//! Naive generalized tensor contraction (`tensordot`) kernels.

use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_common::layout::reshape::layout_reshapeable;

/// `(contracted_a, free_a, contracted_b, free_b)` layouts of a contraction.
pub type SplitTensordotAxes = (Layout<IxD>, Layout<IxD>, Layout<IxD>, Layout<IxD>);

/// Canonical `(M, K)`, `(K, N)`, `(M, N)` layouts for a GEMM contraction.
pub type GemmLayouts = (Layout<Ix2>, Layout<Ix2>, Layout<Ix2>);

/// Split both operands into `(contracted, free)` layouts for the given axes,
/// asserting that the paired contracted shapes agree.
pub fn split_tensordot_axes<DA, DB>(
    la: &Layout<DA>,
    axes_a: &[isize],
    lb: &Layout<DB>,
    axes_b: &[isize],
) -> Result<SplitTensordotAxes>
where
    DA: DimAPI,
    DB: DimAPI,
{
    let (las, lam) = la.dim_split_axes(axes_a)?;
    let (lbs, lbm) = lb.dim_split_axes(axes_b)?;
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axes should be the same"
    )?;
    Ok((las, lam, lbs, lbm))
}

/// Naive tensor contraction `c = tensordot(a, b, (axes_a, axes_b))`.
///
/// `axes_a` and `axes_b` are already normalized non-negative axes, pairwise
/// aligned (`axes_a[i]` of `a` is contracted with `axes_b[i]` of `b`). The
/// output layout `lc` has shape `free(a) ++ free(b)`, the non-contracted axes of
/// `a` (in original order) followed by those of `b`.
pub fn tensordot_naive_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    axes_a: &[isize],
    axes_b: &[isize],
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
{
    let (las, lam, lbs, lbm) = split_tensordot_axes(la, axes_a, lb, axes_b)?;

    let lc = lc.to_dim::<IxD>()?;
    let offset_a = la.offset();
    let offset_b = lb.offset();

    // Free-axis layouts expanded onto the output shape: the block belonging to
    // the *other* operand gets zero stride (a broadcast view). Only read, so
    // aliasing is sound.
    let shape_c = lc.shape().clone();
    let stride_a = lam.stride().iter().copied().chain(core::iter::repeat_n(0isize, lbm.ndim())).collect_vec();
    let stride_b = core::iter::repeat_n(0isize, lam.ndim()).chain(lbm.stride().iter().copied()).collect_vec();
    // SAFETY: both are valid (read-only) broadcast views of real layouts.
    let lam_e = unsafe { Layout::<IxD>::new_unchecked(shape_c.clone(), stride_a, offset_a) };
    let lbm_e = unsafe { Layout::<IxD>::new_unchecked(shape_c, stride_b, offset_b) };

    layout_col_major_dim_dispatch_3(&lc, &lam_e, &lbm_e, |(idx_c, idx_ma, idx_mb)| {
        let mut val_c: TC = Zero::zero();
        let stat = layout_col_major_dim_dispatch_2(&las, &lbs, |(idx_sa, idx_sb)| {
            // `idx_m*` and `idx_s*` each carry the operand offset already; remove
            // the double count once.
            let ia = idx_ma + idx_sa - offset_a;
            let ib = idx_mb + idx_sb - offset_b;
            val_c = val_c.clone() + a[ia].clone() * b[ib].clone();
        });
        stat.rstsr_unwrap();
        c[idx_c].write(val_c);
    })
}

/// If the contraction is expressible as a single 2-D matrix product over
/// copy-free views, return the `(M, K)`, `(K, N)`, `(M, N)` layouts.
///
/// Returns `Ok(None)` when any of the three views would require a copy, or the
/// contracted axes cannot be merged into one `K` factor — the caller then uses
/// the naive kernel. Never materializes data.
pub fn tensordot_gemm_layouts<DA, DB, DC>(
    la: &Layout<DA>,
    axes_a: &[isize],
    lb: &Layout<DB>,
    axes_b: &[isize],
    lc: &Layout<DC>,
    order: FlagOrder,
) -> Result<Option<GemmLayouts>>
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
{
    let (las, lam, lbs, lbm) = split_tensordot_axes(la, axes_a, lb, axes_b)?;
    let lc = lc.to_dim::<IxD>()?;
    let (m, k, n) = (lam.size(), las.size(), lbm.size());

    // Canonical orders: a -> [free..., ctr...], b -> [ctr..., free...].
    let mut shape = lam.shape().clone();
    shape.extend_from_slice(las.shape());
    let mut stride = lam.stride().clone();
    stride.extend_from_slice(las.stride());
    let la_canon = Layout::<IxD>::new(shape, stride, la.offset())?;
    let mut shape = lbs.shape().clone();
    shape.extend_from_slice(lbm.shape());
    let mut stride = lbs.stride().clone();
    stride.extend_from_slice(lbm.stride());
    let lb_canon = Layout::<IxD>::new(shape, stride, lb.offset())?;

    let (Some(la2), Some(lb2), Some(lc2)) = (
        layout_reshapeable(&la_canon, &vec![m, k], order)?,
        layout_reshapeable(&lb_canon, &vec![k, n], order)?,
        layout_reshapeable(&lc, &vec![m, n], order)?,
    ) else {
        return Ok(None);
    };
    Ok(Some((la2.to_dim::<Ix2>()?, lb2.to_dim::<Ix2>()?, lc2.to_dim::<Ix2>()?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_trigger_fires_and_declines() {
        // contiguous (2,3) @ (3,4) -> (2,4), contract a-axis1 with b-axis0
        let la = Layout::<IxD>::new(vec![2, 3], vec![3, 1], 0).unwrap();
        let lb = Layout::<IxD>::new(vec![3, 4], vec![4, 1], 0).unwrap();
        let lc = Layout::<IxD>::new(vec![2, 4], vec![4, 1], 0).unwrap();
        assert!(tensordot_gemm_layouts(&la, &[1], &lb, &[0], &lc, RowMajor).unwrap().is_some());

        // (2,3,4) @ (3,4,5), contract the last two axes of `a` (contiguous
        // block) -> the (M,K) merge is a no-copy view -> GEMM
        let la3 = Layout::<IxD>::new(vec![2, 3, 4], vec![24, 4, 1], 0).unwrap();
        let lb3 = Layout::<IxD>::new(vec![3, 4, 5], vec![20, 5, 1], 0).unwrap();
        let lc3 = Layout::<IxD>::new(vec![2, 5], vec![5, 1], 0).unwrap();
        assert!(tensordot_gemm_layouts(&la3, &[1, 2], &lb3, &[0, 1], &lc3, RowMajor).unwrap().is_some());

        // same shapes but contract `a` axes [0, 2]: the remaining `K` block
        // (strides 24 and 1) is not mergeable without a copy -> naive fallback
        let lb3b = Layout::<IxD>::new(vec![2, 4, 5], vec![20, 5, 1], 0).unwrap();
        let lc3b = Layout::<IxD>::new(vec![3, 5], vec![5, 1], 0).unwrap();
        assert!(tensordot_gemm_layouts(&la3, &[0, 2], &lb3b, &[0, 1], &lc3b, RowMajor).unwrap().is_none());
    }
}
