use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 32;

/// Fold over the manually unrolled `xs` with `f`
///
/// # See also
///
/// This code is from <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>
pub fn unrolled_reduce<TI, TS, I, F, FSum>(mut xs: &[TI], init: I, f: F, f_sum: FSum) -> TS
where
    TI: Clone,
    TS: Clone,
    I: Fn() -> TS,
    F: Fn(TS, TI) -> TS,
    FSum: Fn(TS, TS) -> TS,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        p2 = f(p2, xs[2].clone());
        p3 = f(p3, xs[3].clone());
        p4 = f(p4, xs[4].clone());
        p5 = f(p5, xs[5].clone());
        p6 = f(p6, xs[6].clone());
        p7 = f(p7, xs[7].clone());

        xs = &xs[8..];
    }
    acc = f_sum(acc.clone(), f_sum(p0, p4));
    acc = f_sum(acc.clone(), f_sum(p1, p5));
    acc = f_sum(acc.clone(), f_sum(p2, p6));
    acc = f_sum(acc.clone(), f_sum(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for (i, x) in xs.iter().enumerate() {
        if i >= 7 {
            break;
        }
        acc = f(acc.clone(), x.clone())
    }
    acc
}

pub fn reduce_all_cpu_serial<TI, TS, TO, D, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<D>,
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
) -> Result<TO>
where
    TI: Clone,
    TS: Clone,
    D: DimAPI,
    I: Fn() -> TS,
    F: Fn(TS, TI) -> TS,
    FSum: Fn(TS, TS) -> TS,
    FOut: Fn(TS) -> TO,
{
    // re-align layout
    let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let mut acc = init();
        layout_col_major_dim_dispatch_1(&layout_contig[0], |idx_a| {
            let slc = &a[idx_a..idx_a + size_contig];
            let acc_inner = unrolled_reduce(slc, &init, &f, &f_sum);
            acc = f_sum(acc.clone(), acc_inner);
        })?;
        return Ok(f_out(acc));
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let acc = iter_a.fold(init(), |acc, idx| f(acc, a[idx].clone()));
        return Ok(f_out(acc));
    }
}

pub fn reduce_axes_cpu_serial<TI, TS, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
) -> Result<(Vec<TS>, Layout<IxD>)>
where
    TI: Clone,
    TS: Clone,
    I: Fn() -> TS,
    F: Fn(TS, TI) -> TS,
    FSum: Fn(TS, TS) -> TS,
    FOut: Fn(TS) -> TS,
{
    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;
    let layout_axes = translate_to_col_major_unary(&layout_axes, TensorIterOrder::default())?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // use contiguous reduce_all only when size of contiguous part is large enough
    let (_, size_contig) = translate_to_col_major_with_contig(&[&layout_axes]);
    if size_contig >= CONTIG_SWITCH {
        // generate layouts for actual evaluation
        let layouts_swapped =
            translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // inner layout is axes to be summed
        let mut layout_inner = layout_axes.clone();

        // prepare output
        let len_out = layout_out.size();
        let mut out = unsafe { uninitialized_vec(len_out) };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f, &f_sum, &f_out)?;
                out[idx_out] = acc;
                Ok(())
            },
        )?;
        return Ok((out, layout_out));
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let mut out = vec![init_val; len_out];

        // closure for adding to mutable reference
        let f_add = |a: &mut TS, b: &TI| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_serial(&mut out, &layout_out, a, &layout_inner, f_add)?;
        }
        let fin_inplace = |a: &mut TS| *a = f_out(a.clone());
        op_muta_func_cpu_serial(&mut out, &layout_out, fin_inplace)?;
        return Ok((out, layout_out));
    }
}

pub fn reduce_axes_difftype_cpu_serial<TI, TS, TO, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
) -> Result<(Vec<TO>, Layout<IxD>)>
where
    TI: Clone,
    TS: Clone,
    I: Fn() -> TS,
    F: Fn(TS, TI) -> TS,
    FSum: Fn(TS, TS) -> TS,
    FOut: Fn(TS) -> TO,
{
    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;
    let layout_axes = translate_to_col_major_unary(&layout_axes, TensorIterOrder::default())?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // use contiguous reduce_all only when size of contiguous part is large enough
    let (_, size_contig) = translate_to_col_major_with_contig(&[&layout_axes]);
    if size_contig >= CONTIG_SWITCH {
        // generate layouts for actual evaluation
        let layouts_swapped =
            translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // inner layout is axes to be summed
        let mut layout_inner = layout_axes.clone();

        // prepare output
        let len_out = layout_out.size();
        let mut out = unsafe { uninitialized_vec(len_out) };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f, &f_sum, &f_out)?;
                out[idx_out] = acc;
                Ok(())
            },
        )?;
        return Ok((out, layout_out));
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let mut out = vec![init_val; len_out];

        // closure for adding to mutable reference
        let f_add = |a: &mut TS, b: &TI| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_serial(&mut out, &layout_out, a, &layout_inner, f_add)?;
        }

        let mut out_converted = unsafe { uninitialized_vec(len_out) };
        let f_out = |a: &mut TO, b: &TS| *a = f_out(b.clone());
        op_muta_refb_func_cpu_serial(&mut out_converted, &layout_out, &out, &layout_out, f_out)?;
        return Ok((out_converted, layout_out));
    }
}
