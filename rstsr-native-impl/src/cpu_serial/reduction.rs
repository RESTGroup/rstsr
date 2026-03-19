//! Basic reduction implementation for CPU without parallelization.
//!
//! # Reduction kernel function
//!
//! This module does not involve SIMD optimization. We rely on `unrolled_reduce` and
//! `unrolled_binary_reduce` as non-SIMD but SIMD-aware implementation, leave it to LLVM IR's
//! autovectorization (properly using target=native) to generate SIMD instructions when possible.
//!
//! - `unrolled_reduce` is for reduction with unary input (e.g., sum, mean, max)
//! - `unrolled_binary_reduce` is for reduction with binary input (e.g., dot product, is close)
//!
//! Refer to <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>.
//!
//! # Usual reduction functions
//!
//! Reduction involves 4 functions as arguments:
//! - `init`: initializer
//! - `f`: accumulator + current value -> new accumulator
//! - `f_sum`: accumulator + accumulator -> new accumulator
//! - `f_out`: accumulator -> output
//!
//! To show the relationship, some examples of common reductions are listed below.
//!
//! | reduction | | `init` | `f` | `f_sum` | `f_out` |
//! |---|---|---|---|---|---|
//! | sum | `0` | `acc + x` | `acc1 + acc2` | identity |
//! | mean | `0` | `acc + x` | `acc1 + acc2` | `acc / n` |
//! | max | `T::MIN` | `max(acc, x)` | `max(acc1, acc2)` | identity |
//! | dot | `0` | `acc + x1 * x2` | `acc1 + acc2` | identity |
//! | l2norm | `0` | `acc + x * x` | `acc1 + acc2` | `sqrt(acc)` |
//! | var | `(0, 0)` | `(acc_sum + x, acc_sq + x^2)` | `(acc1_sum + acc2_sum, acc1_sq + acc2_sq)` | `acc_sum_sq / n - (acc_sum / n)^2` |

use crate::prelude_dev::*;
use core::mem::transmute;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 32;

/// Fold over the manually unrolled `xs` with `f`.
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

/// Fold over the manually unrolled `xs1` and `xs2` (binary inputs) with `f`.
///
/// This function does not check that the lengths of `xs1` and `xs2` are the same. The shorter one
/// will determine the number of iterations.
///
/// # See also
///
/// This code is from <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>
pub fn unrolled_binary_reduce<TI1, TI2, TS, I, F, FSum>(
    mut xs1: &[TI1],
    mut xs2: &[TI2],
    init: I,
    f: F,
    f_sum: FSum,
) -> TS
where
    TI1: Clone,
    TI2: Clone,
    TS: Clone,
    I: Fn() -> TS,
    F: Fn(TS, (TI1, TI2)) -> TS,
    FSum: Fn(TS, TS) -> TS,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs1.len() >= 8 && xs2.len() >= 8 {
        p0 = f(p0, (xs1[0].clone(), xs2[0].clone()));
        p1 = f(p1, (xs1[1].clone(), xs2[1].clone()));
        p2 = f(p2, (xs1[2].clone(), xs2[2].clone()));
        p3 = f(p3, (xs1[3].clone(), xs2[3].clone()));
        p4 = f(p4, (xs1[4].clone(), xs2[4].clone()));
        p5 = f(p5, (xs1[5].clone(), xs2[5].clone()));
        p6 = f(p6, (xs1[6].clone(), xs2[6].clone()));
        p7 = f(p7, (xs1[7].clone(), xs2[7].clone()));

        xs1 = &xs1[8..];
        xs2 = &xs2[8..];
    }
    acc = f_sum(acc.clone(), f_sum(p0, p4));
    acc = f_sum(acc.clone(), f_sum(p1, p5));
    acc = f_sum(acc.clone(), f_sum(p2, p6));
    acc = f_sum(acc.clone(), f_sum(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for (i, (x1, x2)) in (xs1.iter().zip(xs2.iter())).enumerate() {
        if i >= 7 {
            break;
        }
        acc = f(acc.clone(), (x1.clone(), x2.clone()))
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
    // Always use K (keep) order for reduction internally (which is the default).
    // Will then translate back to the default order at the end.

    // naming convention
    // - prefix `l`: layout
    // - suffix `s`: [s]ummed (reduced) axes
    // - suffix `m`: re[m]aining axes
    // - suffix `o`: [o]utput
    // - suffix `k`: loc[k]
    // - suffix `c`: [c]ontiguous part
    // - suffix `d`: [d]iscontiguous part

    // create important layouts (summed, remaining)
    let (ls, lm) = la.dim_split_axes(axes)?;

    // summed axes are used together with remaining axes, where offset of layout may double-counted.
    let offset = la.offset();

    // create output layout
    let lo = layout_for_array_copy(&lm, TensorIterOrder::K)?;
    let mut out: Vec<MaybeUninit<TS>> = unsafe { uninitialized_vec(lo.size())? };

    // extract contiguous part and its corresponding dimensions
    // returns: remaining layout, remaining axes loc, contiguous size, contiguous axes loc
    let (as0, _as1, asc, asd) = get_axes_composition(&ls);
    let (am0, _am1, amc, amd) = get_axes_composition(&lm);

    // get some specific sizes of different parts
    let size_s0 = as0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_sc = asc.iter().map(|&i| ls.shape()[i]).product::<usize>();
    let size_m0 = am0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_mc = amc.iter().map(|&i| lm.shape()[i]).product::<usize>();

    if size_sc > 1 {
        // contiguous parts to be summed, call unrolled_reduce for inner reduce
        let amcd = amc.iter().chain(amd.iter()).map(|&i| i as isize).collect_vec();
        let (lmcd, _) = lm.dim_split_axes(&amcd)?;
        let (locd, _) = lo.dim_split_axes(&amcd)?;
        let it_mcd = IterLayoutColMajor::new(&lmcd)?;
        let it_ocd = IterLayoutColMajor::new(&locd)?;

        let asd = asd.iter().map(|&i| i as isize).collect_vec();
        let (lsd, _) = ls.dim_split_axes(&asd)?;
        let it_sd = IterLayoutColMajor::new(&lsd)?;

        it_mcd.zip(it_ocd).for_each(|(i_mcd, i_ocd)| {
            let mut acc = init();
            // handle usual reduction
            it_sd.clone().for_each(|i_sd| {
                let idx_in = i_mcd + i_sd - offset; // double-counted offset
                acc = f_sum(acc.clone(), unrolled_reduce(&a[idx_in..idx_in + size_sc], &init, &f, &f_sum));
            });
            // handle broadcast reduction
            let acc_before = acc.clone();
            for _ in 1..size_s0 {
                acc = f_sum(acc, acc_before.clone());
            }
            out[i_ocd].write(f_out(acc));
        });
    } else if size_mc > 1 {
        // contiguous parts to be remains, but other parts to be summed
        let ascd = asc.iter().chain(asd.iter()).map(|&i| i as isize).collect_vec();
        let (lscd, _) = ls.dim_split_axes(&ascd)?;
        let it_scd = IterLayoutColMajor::new(&lscd)?;

        let amd = amd.iter().map(|&i| i as isize).collect_vec();
        let (lmd, _) = lm.dim_split_axes(&amd)?;
        let (lod, _) = lo.dim_split_axes(&amd)?;
        let it_md = IterLayoutColMajor::new(&lmd)?;
        let it_od = IterLayoutColMajor::new(&lod)?;

        // double check the contigous of output layout
        let amc = amc.iter().map(|&i| i as isize).collect_vec();
        let (loc, _) = lo.dim_split_axes(&amc)?;
        rstsr_assert!(
            loc.f_contig(),
            RuntimeError,
            "probably internal bug: the contiguous part of input must be the same applied to output"
        )?;

        // iterate the discontiguous remain parts
        it_md.zip(it_od).for_each(|(i_md, i_od)| {
            // initialize sequential parts
            let mut vacc = vec![init(); size_mc];
            // iterate the reduction parts
            // - chunk to contiguous output (current chunk size is small, but applicable to most situations)
            const CHUNK: usize = 48;
            vacc.chunks_mut(CHUNK).enumerate().for_each(|(i_chunk, vacc_chunk)| {
                let start = i_chunk * CHUNK;
                let nchunk = vacc_chunk.len();
                it_scd.clone().for_each(|i_scd| {
                    let idx_in = i_md + i_scd - offset; // double-counted offset
                    let slc = &a[idx_in + start..idx_in + start + nchunk];
                    vacc_chunk.iter_mut().zip(slc).for_each(|(acc, x)| {
                        *acc = f(acc.clone(), x.clone());
                    });
                });
            });
            // apply broadcast duplication and finalization function and write to output
            out[i_od..i_od + size_mc].iter_mut().zip(vacc).for_each(|(val, mut acc)| {
                let acc_before = acc.clone();
                for _ in 1..size_s0 {
                    acc = f_sum(acc, acc_before.clone());
                }
                val.write(f_out(acc));
            });
        });
    } else {
        // no contiguous part, just iterate the whole layout with simple fold
        let amd = amd.iter().map(|&i| i as isize).collect_vec();
        let (lmd, _) = lm.dim_split_axes(&amd)?;
        let (lod, _) = lo.dim_split_axes(&amd)?;
        let it_md = IterLayoutColMajor::new(&lmd)?;
        let it_od = IterLayoutColMajor::new(&lod)?;

        let asd = asd.iter().map(|&i| i as isize).collect_vec();
        let (lsd, _) = ls.dim_split_axes(&asd)?;
        let it_sd = IterLayoutColMajor::new(&lsd)?;

        it_md.zip(it_od).for_each(|(i_md, i_od)| {
            let mut acc = it_sd.clone().fold(init(), |acc, i_sd| {
                let idx_in = i_md + i_sd - offset; // double-counted offset
                f(acc, a[idx_in].clone())
            });
            let acc_before = acc.clone();
            for _ in 1..size_s0 {
                acc = f_sum(acc, acc_before.clone());
            }
            out[i_od].write(f_out(acc));
        });
    }

    // Now we handle the broadcast remaining part
    if size_m0 > 1 {
        let am0 = am0.iter().map(|&i| i as isize).collect_vec();
        let (lo0, _) = lo.dim_split_axes(&am0)?;
        let it_o0 = IterLayoutColMajor::new(&lo0)?;

        let amcd = amc.iter().chain(amd.iter()).map(|&i| i as isize).collect_vec();
        let (locd, _) = lo.dim_split_axes(&amcd)?;
        let it_ocd = IterLayoutColMajor::new(&locd)?;

        it_o0.for_each(|idx_o0| {
            it_ocd.clone().for_each(|idx_ocd| {
                let idx_o0 = idx_o0 + idx_ocd - offset; // double-counted offset

                // Safety: the c/d part without broadcast should have been initialized by reduced value
                let val = unsafe { out[idx_ocd].assume_init_read().clone() };
                out[idx_o0].write(val);
            });
        });
    }

    // Safety: all broadcast, discontiguous, contiguous parts have been handled, the `out` is now fully
    // initialized, transmute it to the output type
    let mut out = unsafe { transmute::<Vec<MaybeUninit<TS>>, Vec<TS>>(out) };

    // handle tensor iter order
    if TensorIterOrder::default() != TensorIterOrder::K {
        let lo_default = layout_for_array_copy(&lm, TensorIterOrder::default())?;
        if lo_default != lo {
            let mut out_default: Vec<MaybeUninit<TS>> = unsafe { uninitialized_vec(lo_default.size())? };
            op_muta_refb_func_cpu_serial(&mut out_default, &lo_default, &out, &lo, |a, b| {
                a.write(b.clone());
            })?;
            out = unsafe { transmute::<Vec<MaybeUninit<TS>>, Vec<TS>>(out_default) };
        }
    }

    Ok((out, lo))
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
        let mut out: Vec<MaybeUninit<TO>> = unsafe { uninitialized_vec(len_out)? };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
            unsafe { layout_inner.set_offset(idx_rest) };
            let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f, &f_sum, &f_out)?;
            out[idx_out] = MaybeUninit::new(acc);
            Ok(())
        })?;
        let out = unsafe { transmute::<Vec<MaybeUninit<TO>>, Vec<TO>>(out) };
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

/* #region reduce_binary */

pub fn reduce_all_binary_cpu_serial<TI1, TI2, TS, TO, D, I, F, FSum, FOut>(
    a: &[TI1],
    la: &Layout<D>,
    b: &[TI2],
    lb: &Layout<D>,
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
) -> Result<TO>
where
    TI1: Clone,
    TI2: Clone,
    TS: Clone,
    D: DimAPI,
    I: Fn() -> TS,
    F: Fn(TS, (TI1, TI2)) -> TS,
    FSum: Fn(TS, TS) -> TS,
    FOut: Fn(TS) -> TO,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig >= CONTIG_SWITCH {
        let mut acc = init();
        let la = &layouts_contig[0];
        let lb = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(la, lb, |(idx_a, idx_b)| {
            let slc_a = &a[idx_a..idx_a + size_contig];
            let slc_b = &b[idx_b..idx_b + size_contig];
            let acc_inner = unrolled_binary_reduce(slc_a, slc_b, &init, &f, &f_sum);
            acc = f_sum(acc.clone(), acc_inner);
        })?;
        Ok(f_out(acc))
    } else {
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        let acc =
            izip!(iter_a, iter_b).fold(init(), |acc, (idx_a, idx_b)| f(acc, (a[idx_a].clone(), b[idx_b].clone())));
        Ok(f_out(acc))
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
    let mut out: Vec<MaybeUninit<IxD>> = unsafe { uninitialized_vec(len_out)? };

    // actual evaluation
    izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
        unsafe { layout_inner.set_offset(idx_rest) };
        let acc = reduce_all_unraveled_arg_cpu_serial(a, &layout_inner, &f_comp, &f_eq)?;
        out[idx_out] = MaybeUninit::new(acc);
        Ok(())
    })?;
    let out = unsafe { transmute::<Vec<MaybeUninit<IxD>>, Vec<IxD>>(out) };
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
