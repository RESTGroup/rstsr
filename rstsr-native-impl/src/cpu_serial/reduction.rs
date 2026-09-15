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

pub fn reduce_axes_cpu_serial<TI, TS, TO, I, F, FSum, FOut>(
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
    TO: Clone,
    I: Fn() -> TS,
    F: Fn(TS, TI) -> TS,
    FSum: Fn(TS, TS) -> TS,
    FOut: Fn(TS) -> TO,
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
    // SAFETY (contract): `uninitialized_vec` per trait contract; every `lo.size()`
    // slot is written below exactly once (output offsets from the layout
    // iteration) before the final transmute.
    let mut out: Vec<MaybeUninit<TO>> = unsafe { uninitialized_vec(lo.size())? };

    // extract contiguous part and its corresponding dimensions
    // returns: remaining layout, remaining axes loc, contiguous size, contiguous axes loc
    let (_as1, as0, asc, asd) = get_axes_composition(&ls);
    let (_am1, am0, amc, amd) = get_axes_composition(&lm);

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
            // - chunk to contiguous output (current chunk size is small, but applicable to most
            //   situations)
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

                // Safety: the c/d part without broadcast should have been initialized by reduced
                // value
                // SAFETY: the c/d part without broadcast was initialized by the reduction
                // above (see existing comment).
                let val = unsafe { out[idx_ocd].assume_init_read().clone() };
                out[idx_o0].write(val);
            });
        });
    }

    // Safety: all broadcast, discontiguous, contiguous parts have been handled, the `out` is now
    // fully initialized, transmute it to the output type
    // SAFETY: all `out` elements were written by the reduction above.
    let mut out = unsafe { transmute::<Vec<MaybeUninit<TO>>, Vec<TO>>(out) };

    // handle tensor iter order
    if TensorIterOrder::default() != TensorIterOrder::K {
        let lo_default = layout_for_array_copy(&lm, TensorIterOrder::default())?;
        if lo_default != lo {
            let mut out_default: Vec<MaybeUninit<TO>> = unsafe { uninitialized_vec(lo_default.size())? };
            op_muta_refb_func_cpu_serial(&mut out_default, &lo_default, &out, &lo, |a, b| {
                a.write(b.clone());
            })?;
            // SAFETY: `op_muta_refb_func` above wrote every element of `out_default`.
            out = unsafe { transmute::<Vec<MaybeUninit<TO>>, Vec<TO>>(out_default) };
        }
    }

    Ok((out, lo))
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

/// Comparison and NaN policy for the argmin/argmax-family reduce kernels.
///
/// `Min`/`Max` follow the original rstsr fold semantics: NaN never wins an
/// update, so a NaN at the first scanned position poisons the result and an
/// all-NaN input yields 0. This deliberately DIVERGES from NumPy
/// `np.argmin`/`np.argmax` (which return the first NaN at any position):
/// making NaN win requires an unordered-aware compare per element
/// (`!(x <= best)` needs a second, parity, flag check) or an extra scan
/// pass, either of which de-vectorizes or doubles the cost of this
/// auto-vectorized kernel (measured +55…+100% on small compute-bound
/// inputs, +5…+18% on memory-bound ones). NumPy-style behavior remains
/// available through [`ArgCmp::NanMin`]/[`ArgCmp::NanMax`]
/// (`np.nanargmin`/`np.nanargmax` semantics: NaN elements are skipped, an
/// all-NaN slice raises `InvalidValue`, "All-NaN slice encountered"), which
/// shares the same fast kernel at no measurable cost on NaN-free input.
///
/// Non-floating element types have no NaN and behave identically under all
/// policies. The general closure-based API (`reduce_*_arg_cpu_*`, taking
/// `f_comp`/`f_eq`) is retained alongside for arbitrary (non-standard)
/// comparison/equality semantics; the contiguous 8-lane fast path
/// ([`arg_contig_cpu_serial`]) is only available through this enum — it
/// cannot afford a per-element closure call.
///
/// Note: the `*_arg_cmp_*` signatures are public API of the `rstsr-native-impl`
/// crate (`cpu_serial::reduction` and `cpu_rayon::reduction` are public
/// modules). The tensor-level API in `rstsr-core` is unchanged.
#[derive(Clone, Copy, Debug)]
pub enum ArgCmp {
    Min,
    Max,
    NanMin,
    NanMax,
}

impl ArgCmp {
    /// Whether NaN elements are skipped instead of winning
    /// ([`ArgCmp::NanMin`]/[`ArgCmp::NanMax`]).
    pub fn skip_nan(self) -> bool {
        matches!(self, ArgCmp::NanMin | ArgCmp::NanMax)
    }
}

/// Error message of every all-NaN slice in the `NanMin`/`NanMax` policies
/// (mirrors NumPy's `ValueError("All-NaN slice encountered")`).
pub const ARG_ALL_NAN_MSG: &str = "All-NaN slice encountered";

/// acc-None fallback message of the closure fold (kept from the original
/// `reduce_*_arg_*` implementation).
pub(crate) const FOLD_INVALID_MSG: &str = "reduce_arg seems not returning a valid value.";

/// Block size of the NaN pre-scan for the `Min`/`Max` policies: small enough
/// Contiguous argmin/argmax-family scan; returns the winning flat index by
/// [`ArgCmp`] policy.
///
/// Semantics — identical to a left-to-right fold over an ascending index
/// order with the strict-comparison rule (locked by the T6 correctness gate
/// of the rstsr efficiency campaign):
///
/// - `Min`/`Max`: the first element seeds the accumulator unconditionally; only a strictly smaller
///   (min) / larger (max) value replaces it; ties keep the smaller index; NaN never wins an update,
///   so a NaN at the first scanned position poisons the result to that position's index and an
///   all-NaN input yields 0. Note: this DIVERGES from NumPy `np.argmin`/ `np.argmax` (first NaN at
///   any position wins) — see [`ArgCmp`].
/// - `NanMin`/`NanMax` (NumPy `np.nanargmin`/`np.nanargmax`): NaN elements never enter the
///   accumulators; the seed is the first non-NaN element; an all-NaN input raises `InvalidValue`
///   ([`ARG_ALL_NAN_MSG`]).
///
/// Implementation note: the 8-lane accumulators (ndarray
/// `numeric_util`-style) are seeded by the caller with a guaranteed
/// comparable element; seeding lanes from `xs[0..8]` instead would let a NaN
/// at positions 1..8 block its whole lane (`x > NaN` is false for every
/// later element of that lane).
// `x == x` self-comparisons below are generic NaN checks (false iff NaN);
// the lint's usual "equal operands is a bug" reading does not apply here.
#[allow(clippy::eq_op)]
pub fn arg_contig_cpu_serial<T: Clone + PartialOrd>(xs: &[T], cmp: ArgCmp) -> Result<usize> {
    if cmp.skip_nan() {
        // nanarg policy: seed with the first non-NaN element; all-NaN errors
        let mut seed = None;
        for (i, x) in xs.iter().enumerate() {
            if x == x {
                seed = Some(i);
                break;
            }
        }
        let j = match seed {
            Some(j) => j,
            None => rstsr_raise!(InvalidValue, "{}", ARG_ALL_NAN_MSG)?,
        };
        let (val, idx) = arg_contig_scan_cpu_serial(xs, cmp, &xs[j]);
        let _ = val;
        // nothing strictly beat the first non-NaN element: it is the extremum
        return if idx == usize::MAX { Ok(j) } else { Ok(idx) };
    }
    if !(xs[0] == xs[0]) {
        // NaN first element: nothing can ever strictly beat it -> index 0
        return Ok(0);
    }
    let (val, idx) = arg_contig_scan_cpu_serial(xs, cmp, &xs[0]);
    let _ = val;
    if idx == usize::MAX {
        // nothing strictly beat the seed: the extremum is the first element
        Ok(0)
    } else {
        Ok(idx)
    }
}

/// Core of the contiguous scan: all eight lanes are seeded with `seed`, which
/// must be a comparable (non-NaN) value; the caller owns the NaN policy (the
/// input must not contain NaN that should win). Returns `(best_value, index)`
/// where `index == usize::MAX` means "no element strictly beat `seed`". NaN
/// elements never win an update (their ordered compares are all false); equal
/// values keep the earlier (smaller) index.
pub fn arg_contig_scan_cpu_serial<T: Clone + PartialOrd>(xs: &[T], cmp: ArgCmp, seed: &T) -> (T, usize) {
    const NO_POS: usize = usize::MAX;

    // eight independent (value, index) lanes so that the loop can be kept
    // branch-predictable (LLVM lowers the per-lane updates to cmp+cmov)
    let mut vs: [T; 8] = core::array::from_fn(|_| seed.clone());
    let mut idx: [usize; 8] = [NO_POS; 8];
    let mut chunks = xs.chunks_exact(8);
    match cmp {
        ArgCmp::Max | ArgCmp::NanMax => {
            let mut base = 0;
            for ch in chunks.by_ref() {
                for l in 0..8 {
                    if ch[l] > vs[l] {
                        vs[l] = ch[l].clone();
                        idx[l] = base + l;
                    }
                }
                base += 8;
            }
            for (l, x) in chunks.remainder().iter().enumerate() {
                if x > &vs[l] {
                    vs[l] = x.clone();
                    idx[l] = base + l;
                }
            }
        },
        ArgCmp::Min | ArgCmp::NanMin => {
            let mut base = 0;
            for ch in chunks.by_ref() {
                for l in 0..8 {
                    if ch[l] < vs[l] {
                        vs[l] = ch[l].clone();
                        idx[l] = base + l;
                    }
                }
                base += 8;
            }
            for (l, x) in chunks.remainder().iter().enumerate() {
                if x < &vs[l] {
                    vs[l] = x.clone();
                    idx[l] = base + l;
                }
            }
        },
    }

    // combine lanes: strictly better value wins; on equality the smaller
    // index wins (lane order is not index order); NaN never wins
    let mut best_idx = idx[0];
    let mut best_val = &vs[0];
    for l in 1..8 {
        let better = match cmp {
            ArgCmp::Max | ArgCmp::NanMax => vs[l] > *best_val || (vs[l] == *best_val && idx[l] < best_idx),
            ArgCmp::Min | ArgCmp::NanMin => vs[l] < *best_val || (vs[l] == *best_val && idx[l] < best_idx),
        };
        if better {
            best_idx = idx[l];
            best_val = &vs[l];
        }
    }
    (best_val.clone(), best_idx)
}

/* #region strided fallback comparison closures */

// `x == x` self-comparisons below are generic NaN checks (false iff NaN).

/// Plain argmax comparison (original `reduce_*_arg_*` semantics).
#[inline]
pub(crate) fn f_comp_std_max<T: PartialOrd>(x: Option<T>, y: T) -> Option<bool> {
    match x {
        Some(x) => Some(y > x),
        None => Some(true),
    }
}

/// Plain argmin comparison (mirror of [`f_comp_std_max`]).
#[inline]
pub(crate) fn f_comp_std_min<T: PartialOrd>(x: Option<T>, y: T) -> Option<bool> {
    match x {
        Some(x) => Some(y < x),
        None => Some(true),
    }
}

/// NumPy-nanargmax comparison: NaN current values are skipped entirely.
#[inline]
// `x == x` self-comparison is the generic NaN check (false iff NaN).
#[allow(clippy::eq_op)]
pub(crate) fn f_comp_nan_max<T: PartialOrd>(x: Option<T>, y: T) -> Option<bool> {
    if !(y == y) {
        return None;
    }
    match x {
        Some(x) => Some(y > x),
        None => Some(true),
    }
}

/// NumPy-nanargmin comparison (mirror of [`f_comp_nan_max`]).
#[inline]
// `x == x` self-comparison is the generic NaN check (false iff NaN).
#[allow(clippy::eq_op)]
pub(crate) fn f_comp_nan_min<T: PartialOrd>(x: Option<T>, y: T) -> Option<bool> {
    if !(y == y) {
        return None;
    }
    match x {
        Some(x) => Some(y < x),
        None => Some(true),
    }
}

/// Tie test of the fold: equal values resolve to the smaller index. A NaN
/// current value never ties (the comparison closures filter it beforehand).
#[inline]
pub(crate) fn f_eq_std<T: PartialEq>(x: Option<T>, y: T) -> Option<bool> {
    match x {
        Some(x) => Some(y == x),
        None => Some(false),
    }
}

/* #endregion */

/// Original closure-based fold over [`IndexedIterLayout`] (row-major):
/// implementation core of the general [`reduce_all_unraveled_arg_cpu_serial`]
/// and strided/non-contiguous fallback of
/// [`reduce_all_unraveled_arg_cmp_cpu_serial`]. `invalid_msg` is the error
/// raised when the fold ends without any accepted element (used to give the
/// nanarg policies their "All-NaN slice encountered" message).
#[inline]
fn reduce_all_unraveled_arg_fold_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    f_comp: Fcomp,
    f_eq: Feq,
    invalid_msg: &'static str,
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
        rstsr_raise!(InvalidValue, "{}", invalid_msg)?;
    }
    Ok(acc.unwrap().0)
}

/// Contiguous fast path of [`reduce_all_unraveled_arg_cmp_cpu_serial`].
///
/// Kept out-of-line so that the strided fallback below compiles exactly like
/// the pre-existing closure fold (code-layout hygiene: the fallback must not
/// regress).
#[inline(never)]
fn reduce_all_unraveled_arg_contig_cpu_serial<T, D>(a: &[T], la: &Layout<D>, cmp: ArgCmp) -> Result<D>
where
    T: Clone + PartialOrd,
    D: DimAPI,
{
    // buffer position == row-major visit position for a c-contig layout, so
    // the 8-lane scan reproduces the closure fold exactly (see
    // `arg_contig_cpu_serial` for the semantics contract)
    let offset = la.offset();
    let size = la.size();
    let flat = arg_contig_cpu_serial(&a[offset..offset + size], cmp)?;
    Ok(la.shape().unravel_index_c(flat))
}

/// Argmin/argmax-specialized fast path of
/// [`reduce_all_unraveled_arg_cpu_serial`]: dispatches to the contiguous
/// 8-lane scan when the layout is c-contiguous, and falls back to the
/// original closure fold (comparison direction selected by `cmp`) otherwise.
pub fn reduce_all_unraveled_arg_cmp_cpu_serial<T, D>(a: &[T], la: &Layout<D>, cmp: ArgCmp) -> Result<D>
where
    T: Clone + PartialOrd,
    D: DimAPI,
{
    rstsr_assert!(la.size() > 0, InvalidLayout, "empty sequence is not allowed for reduce_arg.")?;

    if la.c_contig() {
        return reduce_all_unraveled_arg_contig_cpu_serial(a, la, cmp);
    }

    // strided / broadcast fallback: original closure fold, comparison and
    // NaN policy selected by `cmp`
    match cmp {
        ArgCmp::Max => reduce_all_unraveled_arg_fold_cpu_serial(a, la, f_comp_std_max, f_eq_std, FOLD_INVALID_MSG),
        ArgCmp::Min => reduce_all_unraveled_arg_fold_cpu_serial(a, la, f_comp_std_min, f_eq_std, FOLD_INVALID_MSG),
        ArgCmp::NanMax => reduce_all_unraveled_arg_fold_cpu_serial(a, la, f_comp_nan_max, f_eq_std, ARG_ALL_NAN_MSG),
        ArgCmp::NanMin => reduce_all_unraveled_arg_fold_cpu_serial(a, la, f_comp_nan_min, f_eq_std, ARG_ALL_NAN_MSG),
    }
}

/// Argmin/argmax-specialized variant of
/// [`reduce_axes_unraveled_arg_cpu_serial`] (see
/// [`reduce_all_unraveled_arg_cmp_cpu_serial`]).
pub fn reduce_axes_unraveled_arg_cmp_cpu_serial<T, D>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    cmp: ArgCmp,
) -> Result<(Vec<IxD>, Layout<IxD>, Layout<IxD>)>
where
    T: Clone + PartialOrd,
    D: DimAPI,
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
    // SAFETY (contract): every `len_out` slot is written below exactly once
    // (`idx_out` from the layout_out iteration) before the transmute.
    let mut out: Vec<MaybeUninit<IxD>> = unsafe { uninitialized_vec(len_out)? };

    // actual evaluation
    izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
        // SAFETY: `idx_rest` comes from the validated rest-layout iteration; inner
        // layout + offset addresses only in-bounds elements.
        unsafe { layout_inner.set_offset(idx_rest) };
        let acc = reduce_all_unraveled_arg_cmp_cpu_serial(a, &layout_inner, cmp)?;
        out[idx_out] = MaybeUninit::new(acc);
        Ok(())
    })?;
    // SAFETY: all elements written above.
    let out = unsafe { transmute::<Vec<MaybeUninit<IxD>>, Vec<IxD>>(out) };
    // returns (indices, layout_axes, layout_out): each index in `out` is an
    // unraveled position within `layout_axes` (the reduced-axes space), *not*
    // within `layout_out`. Callers that ravel the indices must use
    // `layout_axes.shape()`.
    Ok((out, layout_axes, layout_out))
}

/// Argmin/argmax-specialized variant of [`reduce_all_arg_cpu_serial`] (see
/// [`reduce_all_unraveled_arg_cmp_cpu_serial`]).
pub fn reduce_all_arg_cmp_cpu_serial<T, D>(a: &[T], la: &Layout<D>, cmp: ArgCmp, order: FlagOrder) -> Result<usize>
where
    T: Clone + PartialOrd,
    D: DimAPI,
{
    let idx = reduce_all_unraveled_arg_cmp_cpu_serial(a, la, cmp)?;
    let pseudo_shape = la.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    Ok(pseudo_layout.index_uncheck(idx.as_ref()) as usize)
}

/// Argmin/argmax-specialized variant of [`reduce_axes_arg_cpu_serial`] (see
/// [`reduce_all_unraveled_arg_cmp_cpu_serial`]).
pub fn reduce_axes_arg_cmp_cpu_serial<T, D>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    cmp: ArgCmp,
    order: FlagOrder,
) -> Result<(Vec<usize>, Layout<IxD>)>
where
    T: Clone + PartialOrd,
    D: DimAPI,
{
    let (idx, layout_axes, layout) = reduce_axes_unraveled_arg_cmp_cpu_serial(a, la, axes, cmp)?;
    // each index in `idx` is an unraveled position within the reduced-axes space
    // (`layout_axes`), so the raveling pseudo-layout must use `layout_axes.shape()`,
    // not the output layout's shape. Using the output shape here indexed out of
    // bounds for ndim >= 3 (the reduced space has rank 1 but the output has rank
    // ndim - 1).
    let pseudo_shape = layout_axes.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    let out = idx.into_iter().map(|x| pseudo_layout.index_uncheck(x.as_ref()) as usize).collect();
    Ok((out, layout))
}

// General closure-based arg-reduction API (restored): kept for arbitrary
// comparison/equality semantics, e.g. future non-standard arg-reductions.
// argmin/argmax call sites use the `*_arg_cmp_*` specializations above, which
// share the same fold core for the strided/non-contiguous path.

/// General closure-based arg-reduction over all axes, unraveled index output.
/// `f_comp(acc, cur)` decides whether `cur` is accepted (`Some(true)`),
/// `f_eq(acc, cur)` whether it ties (smaller index then wins); `None` from
/// either skips the element. For plain argmin/argmax prefer the specialized
/// [`reduce_all_unraveled_arg_cmp_cpu_serial`].
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
    reduce_all_unraveled_arg_fold_cpu_serial(a, la, f_comp, f_eq, FOLD_INVALID_MSG)
}

/// General closure-based arg-reduction over given axes, unraveled index
/// output. For plain argmin/argmax prefer
/// [`reduce_axes_unraveled_arg_cmp_cpu_serial`].
pub fn reduce_axes_unraveled_arg_cpu_serial<T, D, Fcomp, Feq>(
    a: &[T],
    la: &Layout<D>,
    axes: &[isize],
    f_comp: Fcomp,
    f_eq: Feq,
) -> Result<(Vec<IxD>, Layout<IxD>, Layout<IxD>)>
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
    // SAFETY (contract): every `len_out` slot is written below exactly once
    // (`idx_out` from the layout_out iteration) before the transmute.
    let mut out: Vec<MaybeUninit<IxD>> = unsafe { uninitialized_vec(len_out)? };

    // actual evaluation
    izip!(iter_out_swapped, iter_rest_swapped).try_for_each(|(idx_out, idx_rest)| -> Result<()> {
        // SAFETY: `idx_rest` comes from the validated rest-layout iteration; inner
        // layout + offset addresses only in-bounds elements.
        unsafe { layout_inner.set_offset(idx_rest) };
        let acc = reduce_all_unraveled_arg_cpu_serial(a, &layout_inner, &f_comp, &f_eq)?;
        out[idx_out] = MaybeUninit::new(acc);
        Ok(())
    })?;
    // SAFETY: all elements written above.
    let out = unsafe { transmute::<Vec<MaybeUninit<IxD>>, Vec<IxD>>(out) };
    // returns (indices, layout_axes, layout_out): each index in `out` is an
    // unraveled position within `layout_axes` (the reduced-axes space), *not*
    // within `layout_out`. Callers that ravel the indices must use
    // `layout_axes.shape()`.
    Ok((out, layout_axes, layout_out))
}

/// General closure-based arg-reduction over all axes, raveled index output.
/// For plain argmin/argmax prefer [`reduce_all_arg_cmp_cpu_serial`].
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
    Ok(pseudo_layout.index_uncheck(idx.as_ref()) as usize)
}

/// General closure-based arg-reduction over given axes, raveled index output.
/// For plain argmin/argmax prefer [`reduce_axes_arg_cmp_cpu_serial`].
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
    let (idx, layout_axes, layout) = reduce_axes_unraveled_arg_cpu_serial(a, la, axes, f_comp, f_eq)?;
    // each index in `idx` is an unraveled position within the reduced-axes space
    // (`layout_axes`), so the raveling pseudo-layout must use `layout_axes.shape()`,
    // not the output layout's shape. Using the output shape here indexed out of
    // bounds for ndim >= 3 (the reduced space has rank 1 but the output has rank
    // ndim - 1).
    let pseudo_shape = layout_axes.shape();
    let pseudo_layout = match order {
        RowMajor => pseudo_shape.c(),
        ColMajor => pseudo_shape.f(),
    };
    let out = idx.into_iter().map(|x| pseudo_layout.index_uncheck(x.as_ref()) as usize).collect();
    Ok((out, layout))
}

/* #endregion */
