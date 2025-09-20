use crate::prelude_dev::*;
use core::mem::transmute;

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

/* #region reduce */

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
        Ok(f_out(acc))
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let acc = iter_a.fold(init(), |acc, idx| f(acc, a[idx].clone()));
        Ok(f_out(acc))
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
        let layouts_swapped = translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // inner layout is axes to be summed
        let mut layout_inner = layout_axes.clone();

        // prepare output
        let len_out = layout_out.size();
        let mut out = unsafe { uninitialized_vec(len_out)? };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
            unsafe { layout_inner.set_offset(idx_rest) };
            let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f, &f_sum, &f_out)?;
            out[idx_out] = acc;
            Ok(())
        })?;
        Ok((out, layout_out))
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let out = vec![init_val; len_out];
        let mut out = unsafe { transmute::<Vec<TS>, Vec<MaybeUninit<TS>>>(out) };

        // closure for adding to mutable reference
        let f_add = |a: &mut MaybeUninit<TS>, b: &TI| unsafe {
            a.write(f(a.assume_init_read(), b.clone()));
        };

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_serial(&mut out, &layout_out, a, &layout_inner, f_add)?;
        }
        let fin_inplace = |a: &mut MaybeUninit<TS>| unsafe {
            a.write(f_out(a.assume_init_read()));
        };
        op_muta_func_cpu_serial(&mut out, &layout_out, fin_inplace)?;
        let out = unsafe { transmute::<Vec<MaybeUninit<TS>>, Vec<TS>>(out) };
        Ok((out, layout_out))
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
        let layouts_swapped = translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // inner layout is axes to be summed
        let mut layout_inner = layout_axes.clone();

        // prepare output
        let len_out = layout_out.size();
        let mut out = unsafe { uninitialized_vec(len_out)? };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
            unsafe { layout_inner.set_offset(idx_rest) };
            let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f, &f_sum, &f_out)?;
            out[idx_out] = acc;
            Ok(())
        })?;
        Ok((out, layout_out))
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let out = vec![init_val; len_out];
        let mut out = unsafe { transmute::<Vec<TS>, Vec<MaybeUninit<TS>>>(out) };

        // closure for adding to mutable reference
        let f_add = |a: &mut MaybeUninit<TS>, b: &TI| unsafe {
            a.write(f(a.assume_init_read(), b.clone()));
        };

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_serial(&mut out, &layout_out, a, &layout_inner, f_add)?;
        }

        let mut out_converted = unsafe { uninitialized_vec(len_out)? };
        let f_out = |a: &mut MaybeUninit<TO>, b: &MaybeUninit<TS>| unsafe {
            a.write(f_out(b.assume_init_read()));
        };
        op_muta_refb_func_cpu_serial(&mut out_converted, &layout_out, &out, &layout_out, f_out)?;
        let out_converted = unsafe { transmute::<Vec<MaybeUninit<TO>>, Vec<TO>>(out_converted) };
        Ok((out_converted, layout_out))
    }
}

/* #endregion */

/* #region reduce unraveled axes */

pub fn reduce_all_unraveled_arg_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    f_comp: Fcomp,
    f_eq: Feq,
) -> Result<D>
where
    T: Clone,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool>,
    Feq: Fn(Option<T>, T) -> Option<bool>,
{
    rstsr_assert!(la.size() > 0, InvalidLayout, "empty sequence is not allowed for reduce_arg.")?;

    let fold_func = |acc: Option<(D, T)>, (cur_idx, cur_offset): (D, usize)| {
        let cur_val = a[cur_offset].clone();

        let comp = f_comp(acc.as_ref().map(|(_, val)| val.clone()), cur_val.clone());
        if let Some(comp) = comp {
            if comp {
                // cond 1: current value is accepted
                Some((cur_idx, cur_val))
            } else {
                let comp_eq = f_eq(acc.as_ref().map(|(_, val)| val.clone()), cur_val.clone());
                if comp_eq.is_some_and(|x| x) {
                    // cond 2: current value is same with previous value, return smaller index
                    if let Some(acc_idx) = acc.as_ref().map(|(idx, _)| idx.clone()) {
                        if cur_idx < acc_idx {
                            Some((cur_idx, cur_val))
                        } else {
                            acc
                        }
                    } else {
                        Some((cur_idx, cur_val))
                    }
                } else {
                    // cond 3: current value is not accepted
                    acc
                }
            }
        } else {
            // cond 4: current comparasion is not valid
            acc
        }
    };

    let iter_a = IndexedIterLayout::new(la, RowMajor)?;
    let acc = iter_a.into_iter().fold(None, fold_func);
    if acc.is_none() {
        rstsr_raise!(InvalidValue, "reduce_arg seems not returning a valid value.")?;
    }
    Ok(acc.unwrap().0)
}

pub fn reduce_axes_unraveled_arg_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    f_comp: Fcomp,
    f_eq: Feq,
) -> Result<(Vec<IxD>, Layout<IxD>)>
where
    T: Clone,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool>,
    Feq: Fn(Option<T>, T) -> Option<bool>,
{
    rstsr_assert!(la.size() > 0, InvalidLayout, "empty sequence is not allowed for reduce_arg.")?;

    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // generate layouts for actual evaluation
    let layouts_swapped = translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
    let layout_out_swapped = &layouts_swapped[0];
    let layout_rest_swapped = &layouts_swapped[1];

    // iterate both layout_rest and layout_out
    let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
    let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

    // inner layout is axes to be summed
    let mut layout_inner = layout_axes.clone();

    // prepare output
    let len_out = layout_out.size();
    let mut out = vec![IxD::default(); len_out];

    // actual evaluation
    izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
        unsafe { layout_inner.set_offset(idx_rest) };
        let acc = reduce_all_unraveled_arg_cpu_serial(a, &layout_inner, &f_comp, &f_eq)?;
        out[idx_out] = acc;
        Ok(())
    })?;
    Ok((out, layout_out))
}

pub fn reduce_all_arg_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    f_comp: Fcomp,
    f_eq: Feq,
    order: FlagOrder,
) -> Result<usize>
where
    T: Clone,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool>,
    Feq: Fn(Option<T>, T) -> Option<bool>,
{
    let idx = reduce_all_unraveled_arg_cpu_serial(a, la, f_comp, f_eq)?;
    let pseudo_shape = la.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    unsafe { Ok(pseudo_layout.index_uncheck(idx.as_ref()) as usize) }
}

pub fn reduce_axes_arg_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    f_comp: Fcomp,
    f_eq: Feq,
    order: FlagOrder,
) -> Result<(Vec<usize>, Layout<IxD>)>
where
    T: Clone,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool>,
    Feq: Fn(Option<T>, T) -> Option<bool>,
{
    let (idx, layout) = reduce_axes_unraveled_arg_cpu_serial(a, la, axes, f_comp, f_eq)?;
    let pseudo_shape = layout.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    let out = idx.into_iter().map(|x| unsafe { pseudo_layout.index_uncheck(x.as_ref()) as usize }).collect();
    Ok((out, layout))
}

/* #endregion */
