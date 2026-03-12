use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_dtype_traits::ExtNum;

pub fn vecdot_naive_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    axis: isize,
    order: FlagOrder,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TA: ExtNum,
{
    // check axis
    let (axis_a, axis_b) = if axis < 0 {
        rstsr_pattern!(
            axis,
            -(la.ndim().min(lb.ndim()) as isize)..=-1,
            InvalidValue,
            "axis should be [-N, -1] where N is min(a.ndim, b.ndim)"
        )?;
        let axis_a = (axis + la.ndim() as isize) as usize;
        let axis_b = (axis + lb.ndim() as isize) as usize;
        (axis_a, axis_b)
    } else {
        rstsr_pattern!(
            axis,
            0..(la.ndim().min(lb.ndim()) as isize),
            InvalidValue,
            "axis should be [0, N) where N is min(a.ndim, b.ndim)"
        )?;
        (axis as usize, axis as usize)
    };
    rstsr_assert_eq!(
        la.shape()[axis_a],
        lb.shape()[axis_b],
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;

    // get iterator over lc, and chopped la and lb
    let la_chop = la.to_dim::<IxD>()?.dim_chop(axis)?;
    let lb_chop = lb.to_dim::<IxD>()?.dim_chop(axis)?;
    let lc = lc.to_dim::<IxD>()?;
    // broadcast layouts for computation
    let (lc_b, la_chop) = broadcast_layout_to_first(&lc, &la_chop, order)?;
    let (lc_b, lb_chop) = broadcast_layout_to_first(&lc_b, &lb_chop, order)?;
    rstsr_assert_eq!(lc_b, lc, InvalidLayout, "layout of c seems not broadcasted from a or b after axis sum")?;

    // stride for the contracted axis
    let step_a = la.stride()[axis_a];
    let step_b = lb.stride()[axis_b];
    let unit_step = step_a == step_b && step_a == 1;

    // number of elements to be contracted
    let n_contract = la.shape()[axis_a];

    // broadcast layouts for computation
    let layouts = translate_to_col_major(&[&lc, &la_chop, &lb_chop], TensorIterOrder::K)?;
    let lc = &layouts[0];
    let la_chop = &layouts[1];
    let lb_chop = &layouts[2];

    // stride 1 can benefit from faster reduction
    if unit_step {
        layout_col_major_dim_dispatch_3(lc, la_chop, lb_chop, |(idx_c, idx_a, idx_b)| {
            let val_c = unrolled_binary_reduce(
                &a[idx_a..idx_a + n_contract],
                &b[idx_b..idx_b + n_contract],
                || Zero::zero(),
                |acc, (x, y)| acc + x.ext_conj() * y,
                |acc, v| acc + v,
            );
            c[idx_c].write(val_c);
        })
    } else {
        layout_col_major_dim_dispatch_3(lc, la_chop, lb_chop, |(idx_c, idx_a, idx_b)| {
            let val_c = (0..n_contract).fold(Zero::zero(), |acc, i| {
                let x = &a[(idx_a as isize + i as isize * step_a) as usize];
                let y = &b[(idx_b as isize + i as isize * step_b) as usize];
                acc + x.clone().ext_conj() * y.clone()
            });
            c[idx_c].write(val_c);
        })
    }
}
