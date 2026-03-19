use crate::prelude_dev::*;
use core::mem::transmute;
use core::sync::atomic::{AtomicPtr, Ordering};
use rayon::prelude::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 32;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
// 64 kB for f64
const PARALLEL_SWITCH: usize = 1024;
// This value is the maximum chunk in parallel iteration.
const PARALLEL_CHUNK_MAX: usize = 1024;
// Currently, we do not make contiguous parts to be parallel. Only outer
// iteration is parallelized.

/* #region reduce */

pub fn reduce_all_cpu_rayon<TI, TS, TO, D, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<D>,
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: Option<&ThreadPool>,
) -> Result<TO>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    TO: Clone + Send + Sync,
    D: DimAPI,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_all_cpu_serial(a, la, init, f, f_sum, f_out);
    }

    // re-align layout
    let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let task = || {
                iter_a
                    .into_par_iter()
                    .fold(&init, |acc_inner, idx_a| {
                        let slc = &a[idx_a..idx_a + size_contig];
                        f_sum(acc_inner, unrolled_reduce(slc, &init, &f, &f_sum))
                    })
                    .reduce(&init, &f_sum)
            };
            let acc = match pool {
                None => task(),
                Some(pool) => pool.install(task),
            };
            Ok(f_out(acc))
        } else {
            // parallel inner iteration
            let chunk = PARALLEL_CHUNK_MAX;
            let task = || {
                iter_a
                    .into_par_iter()
                    .fold(&init, |acc_inner, idx_a| {
                        let res = (0..size_contig)
                            .into_par_iter()
                            .step_by(chunk)
                            .fold(&init, |acc_chunk, idx| {
                                let chunk = chunk.min(size_contig - idx);
                                let start = idx_a + idx;
                                let slc = &a[start..start + chunk];
                                f_sum(acc_chunk, unrolled_reduce(slc, &init, &f, &f_sum))
                            })
                            .reduce(&init, &f_sum);
                        f_sum(acc_inner, res)
                    })
                    .reduce(&init, &f_sum)
            };
            let acc = match pool {
                None => task(),
                Some(pool) => pool.install(task),
            };
            Ok(f_out(acc))
        }
    } else {
        // manual fold when not contiguous
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let task = || iter_a.into_par_iter().fold(&init, |acc, idx| f(acc, a[idx].clone())).reduce(&init, &f_sum);
        let acc = match pool {
            None => task(),
            Some(pool) => pool.install(task),
        };
        Ok(f_out(acc))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reduce_axes_cpu_rayon<TI, TS, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: Option<&ThreadPool>,
) -> Result<(Vec<TS>, Layout<IxD>)>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TS + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_axes_cpu_serial(a, la, axes, init, f, f_sum, f_out);
    }

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
    let (_as1, as0, asc, asd) = get_axes_composition(&ls);
    let (_am1, am0, amc, amd) = get_axes_composition(&lm);

    // get some specific sizes of different parts
    let size_s0 = as0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_sc = asc.iter().map(|&i| ls.shape()[i]).product::<usize>();
    let size_m0 = am0.iter().map(|&i| lm.shape()[i]).product::<usize>();
    let size_mc = amc.iter().map(|&i| lm.shape()[i]).product::<usize>();

    let mut task = || -> Result<()> {
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

            it_mcd.into_par_iter().zip(it_ocd).for_each(|(i_mcd, i_ocd)| {
                let mut acc = it_sd
                    .clone()
                    .into_par_iter()
                    .fold(&init, |acc, i_sd| {
                        let idx_in = i_mcd + i_sd - offset; // double-counted offset
                        f_sum(acc, unrolled_reduce(&a[idx_in..idx_in + size_sc], &init, &f, &f_sum))
                    })
                    .reduce(&init, &f_sum);
                // handle broadcast reduction
                let acc_before = acc.clone();
                for _ in 1..size_s0 {
                    acc = f_sum(acc, acc_before.clone());
                }
                unsafe {
                    let ptr_out_ocd = out.as_ptr().add(i_ocd) as *mut MaybeUninit<TS>;
                    (*ptr_out_ocd).write(f_out(acc));
                }
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
            it_md.into_par_iter().zip(it_od).for_each(|(i_md, i_od)| {
                // initialize sequential parts
                let mut vacc = vec![init(); size_mc];
                // iterate the reduction parts
                // - chunk to contiguous output (current chunk size is small, but applicable to most situations)
                // - sequential iteration in chunks for reduction (parallel it can lead to racing)
                const CHUNK: usize = 64;
                vacc.par_chunks_mut(CHUNK).enumerate().for_each(|(i_chunk, vacc_chunk)| {
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
                (0..size_mc).into_par_iter().for_each(|i_mc| unsafe {
                    let ptr_out = out.as_ptr().add(i_od + i_mc) as *mut MaybeUninit<TS>;
                    let mut acc = vacc[i_mc].clone();
                    for _ in 1..size_s0 {
                        acc = f_sum(acc, vacc[i_mc].clone());
                    }
                    (*ptr_out).write(f_out(acc));
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

            it_md.into_par_iter().zip(it_od).for_each(|(i_md, i_od)| {
                let mut acc = it_sd
                    .clone()
                    .into_par_iter()
                    .fold(&init, |acc, i_sd| {
                        let idx_in = i_md + i_sd - offset; // double-counted offset
                        f(acc, a[idx_in].clone())
                    })
                    .reduce(&init, &f_sum);
                let acc_before = acc.clone();
                for _ in 1..size_s0 {
                    acc = f_sum(acc, acc_before.clone());
                }
                unsafe {
                    let ptr_out_od = out.as_ptr().add(i_od) as *mut MaybeUninit<TS>;
                    (*ptr_out_od).write(f_out(acc));
                }
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

        Ok(())
    };

    match pool {
        None => task()?,
        Some(pool) => pool.install(task)?,
    };

    // Safety: all broadcast, discontiguous, contiguous parts have been handled, the `out` is now fully
    // initialized, transmute it to the output type
    let mut out = unsafe { transmute::<Vec<MaybeUninit<TS>>, Vec<TS>>(out) };

    // handle tensor iter order
    if TensorIterOrder::default() != TensorIterOrder::K {
        let lo_default = layout_for_array_copy(&lm, TensorIterOrder::default())?;
        if lo_default != lo {
            let mut out_default: Vec<MaybeUninit<TS>> = unsafe { uninitialized_vec(lo_default.size())? };
            let mut func = |a: &mut MaybeUninit<TS>, b: &TS| {
                a.write(b.clone());
            };
            op_muta_refb_func_cpu_rayon(&mut out_default, &lo_default, &out, &lo, &mut func, pool)?;
            out = unsafe { transmute::<Vec<MaybeUninit<TS>>, Vec<TS>>(out_default) };
        }
    }

    Ok((out, lo))
}

#[allow(clippy::too_many_arguments)]
pub fn reduce_axes_difftype_cpu_rayon<TI, TS, TO, I, F, FSum, FOut>(
    a: &[TI],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: Option<&ThreadPool>,
) -> Result<(Vec<TO>, Layout<IxD>)>
where
    TI: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    TO: Clone + Send + Sync,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, TI) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_axes_difftype_cpu_serial(a, la, axes, init, f, f_sum, f_out);
    }

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

        // prepare output
        let len_out = layout_out.size();
        let mut out: Vec<MaybeUninit<TO>> = unsafe { uninitialized_vec(len_out)? };
        let out_ptr = AtomicPtr::new(out.as_mut_ptr());

        // actual evaluation
        let task = || {
            (iter_out_swapped, iter_rest_swapped).into_par_iter().try_for_each(|(idx_out, idx_rest)| -> Result<()> {
                let out_ptr = out_ptr.load(Ordering::Relaxed);
                // let out_ptr = out_ptr.get();
                let mut layout_inner = layout_axes.clone();
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_rayon(a, &layout_inner, &init, &f, &f_sum, &f_out, pool)?;
                unsafe { *out_ptr.add(idx_out) = MaybeUninit::new(acc) };
                Ok(())
            })
        };
        match pool {
            None => task()?,
            Some(pool) => pool.install(task)?,
        };
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
        let mut f_add = |a: &mut MaybeUninit<TS>, b: &TI| unsafe {
            a.write(f(a.assume_init_read(), b.clone()));
        };

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_rayon(&mut out, &layout_out, a, &layout_inner, &mut f_add, pool)?;
        }

        let mut out_converted = unsafe { uninitialized_vec(len_out)? };
        let mut f_out = |a: &mut MaybeUninit<TO>, b: &MaybeUninit<TS>| unsafe {
            a.write(f_out(b.assume_init_read()));
        };
        op_muta_refb_func_cpu_rayon(&mut out_converted, &layout_out, &out, &layout_out, &mut f_out, pool)?;
        let out_converted = unsafe { transmute::<Vec<MaybeUninit<TO>>, Vec<TO>>(out_converted) };
        Ok((out_converted, layout_out))
    }
}

/* #endregion */

/* #region reduce_binary */

pub fn reduce_all_binary_cpu_rayon<TI1, TI2, TS, TO, D, I, F, FSum, FOut>(
    a: &[TI1],
    la: &Layout<D>,
    b: &[TI2],
    lb: &Layout<D>,
    init: I,
    f: F,
    f_sum: FSum,
    f_out: FOut,
    pool: Option<&ThreadPool>,
) -> Result<TO>
where
    TI1: Clone + Send + Sync,
    TI2: Clone + Send + Sync,
    TS: Clone + Send + Sync,
    TO: Clone + Send + Sync,
    D: DimAPI,
    I: Fn() -> TS + Send + Sync,
    F: Fn(TS, (TI1, TI2)) -> TS + Send + Sync,
    FSum: Fn(TS, TS) -> TS + Send + Sync,
    FOut: Fn(TS) -> TO + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_all_binary_cpu_serial(a, la, b, lb, init, f, f_sum, f_out);
    }

    // re-allign layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let iter_a = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_contig[1])?;
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let task = || {
                (iter_a, iter_b)
                    .into_par_iter()
                    .fold(&init, |acc_inner, (idx_a, idx_b)| {
                        let slc_a = &a[idx_a..idx_a + size_contig];
                        let slc_b = &b[idx_b..idx_b + size_contig];
                        f_sum(acc_inner, unrolled_binary_reduce(slc_a, slc_b, &init, &f, &f_sum))
                    })
                    .reduce(&init, &f_sum)
            };
            let acc = match pool {
                None => task(),
                Some(pool) => pool.install(task),
            };
            Ok(f_out(acc))
        } else {
            // parallel inner iteration
            let chunk = PARALLEL_CHUNK_MAX;
            let task = || {
                (iter_a, iter_b)
                    .into_par_iter()
                    .fold(&init, |acc_inner, (idx_a, idx_b)| {
                        let res = (0..size_contig)
                            .into_par_iter()
                            .step_by(chunk)
                            .fold(&init, |acc_chunk, idx| {
                                let chunk = chunk.min(size_contig - idx);
                                let start_a = idx_a + idx;
                                let start_b = idx_b + idx;
                                let slc_a = &a[start_a..start_a + chunk];
                                let slc_b = &b[start_b..start_b + chunk];
                                f_sum(acc_chunk, unrolled_binary_reduce(slc_a, slc_b, &init, &f, &f_sum))
                            })
                            .reduce(&init, &f_sum);
                        f_sum(acc_inner, res)
                    })
                    .reduce(&init, &f_sum)
            };
            let acc = match pool {
                None => task(),
                Some(pool) => pool.install(task),
            };
            Ok(f_out(acc))
        }
    } else {
        // manual fold when not contiguous
        let iter_a = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_b = IterLayoutColMajor::new(&layouts_full[1])?;
        let task = || {
            (iter_a, iter_b)
                .into_par_iter()
                .fold(&init, |acc, (idx_a, idx_b)| f(acc, (a[idx_a].clone(), b[idx_b].clone())))
                .reduce(&init, &f_sum)
        };
        let acc = match pool {
            None => task(),
            Some(pool) => pool.install(task),
        };
        Ok(f_out(acc))
    }
}

/* #endregion */

/* #region reduce unraveled axes */

pub fn reduce_all_unraveled_arg_cpu_rayon<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    f_comp: Fcomp,
    f_eq: Feq,
    pool: Option<&ThreadPool>,
) -> Result<D>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
    Feq: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
{
    rstsr_assert!(la.size() > 0, InvalidLayout, "empty sequence is not allowed for reduce_arg.")?;

    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_all_unraveled_arg_cpu_serial(a, la, f_comp, f_eq);
    }

    let fold_func = |acc: Option<(D, T)>, (cur_idx, cur_offset): (D, usize)| -> Option<(D, T)> {
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
    let sum_func = |acc1: Option<(D, T)>, acc2: Option<(D, T)>| match (acc1, acc2) {
        (Some((idx1, val1)), Some((idx2, _))) => {
            fold_func(Some((idx1, val1)), (idx2.clone(), unsafe { la.index_uncheck(idx2.as_ref()) as usize }))
        },
        (Some((idx1, val1)), None) => Some((idx1, val1)),
        (None, Some((idx2, val2))) => Some((idx2, val2)),
        (None, None) => None,
    };

    let iter_a = IndexedIterLayout::new(la, RowMajor)?;
    let task = || iter_a.into_par_iter().fold(|| None, fold_func).reduce(|| None, sum_func);
    let acc = match pool {
        None => task(),
        Some(pool) => pool.install(task),
    };
    if acc.is_none() {
        rstsr_raise!(InvalidValue, "reduce_arg seems not returning a valid value.")?;
    }
    Ok(acc.unwrap().0)
}

pub fn reduce_axes_unraveled_arg_cpu_rayon<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    f_comp: Fcomp,
    f_eq: Feq,
    pool: Option<&ThreadPool>,
) -> Result<(Vec<IxD>, Layout<IxD>)>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
    Feq: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH {
        return reduce_axes_unraveled_arg_cpu_serial(a, la, axes, f_comp, f_eq);
    }

    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;
    let layout_axes = translate_to_col_major_unary(&layout_axes, TensorIterOrder::default())?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // generate layouts for actual evaluation
    let layouts_swapped = translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
    let layout_out_swapped = &layouts_swapped[0];
    let layout_rest_swapped = &layouts_swapped[1];

    // iterate both layout_rest and layout_out
    let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
    let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

    // prepare output
    let len_out = layout_out.size();
    let mut out: Vec<MaybeUninit<IxD>> = unsafe { uninitialized_vec(len_out)? };
    let out_ptr = AtomicPtr::new(out.as_mut_ptr());

    // actual evaluation
    let task = || {
        (iter_out_swapped, iter_rest_swapped).into_par_iter().try_for_each(|(idx_out, idx_rest)| -> Result<()> {
            let out_ptr = out_ptr.load(Ordering::Relaxed);
            // let out_ptr = out_ptr.get();
            let mut layout_inner = layout_axes.clone();
            unsafe { layout_inner.set_offset(idx_rest) };
            let acc = reduce_all_unraveled_arg_cpu_rayon(a, &layout_inner, &f_comp, &f_eq, pool)?;
            unsafe { *out_ptr.add(idx_out) = MaybeUninit::new(acc) };
            Ok(())
        })
    };
    match pool {
        None => task()?,
        Some(pool) => pool.install(task)?,
    };
    let out = unsafe { transmute::<Vec<MaybeUninit<IxD>>, Vec<IxD>>(out) };
    Ok((out, layout_out))
}

pub fn reduce_all_arg_cpu_rayon<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    f_comp: Fcomp,
    f_eq: Feq,
    order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<usize>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
    Feq: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
{
    let idx = reduce_all_unraveled_arg_cpu_rayon(a, la, f_comp, f_eq, pool)?;
    let pseudo_shape = la.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    unsafe { Ok(pseudo_layout.index_uncheck(idx.as_ref()) as usize) }
}

pub fn reduce_axes_arg_cpu_rayon<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    f_comp: Fcomp,
    f_eq: Feq,
    order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<(Vec<usize>, Layout<IxD>)>
where
    T: Clone + Send + Sync,
    D: DimAPI,
    Fcomp: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
    Feq: Fn(Option<T>, T) -> Option<bool> + Send + Sync,
{
    let (idx, layout) = reduce_axes_unraveled_arg_cpu_rayon(a, la, axes, f_comp, f_eq, pool)?;
    let pseudo_shape = layout.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    let task = || idx.into_par_iter().map(|x| unsafe { pseudo_layout.index_uncheck(x.as_ref()) as usize }).collect();
    let out = match pool {
        None => task(),
        Some(pool) => pool.install(task),
    };
    Ok((out, layout))
}

/* #endregion */
