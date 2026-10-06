//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

// Minimum problem size for the blocked 2-D iteration path; smaller problems
// keep the generic layout-iterator path.
const TILE_SWITCH: usize = 4096;
// Tile edge (elements). Within a [TILE, TILE] tile, each layout's inner-axis
// hop stays within TILE cache lines (L1-resident) for the whole tile pass,
// instead of paging through the entire operand with a ~16 KiB stride.
const TILE: usize = 64;

/// Whether the blocked 2-D path applies: no common f-contiguous prefix was
/// found (caller checks `size_contig < CONTIG_SWITCH`), the problem is 2-D
/// and large enough, and **no layout has a broadcast axis (shape > 1 with
/// stride 0) on any axis** — broadcast operands keep the generic iterator
/// path. This guard checks *all* participating layouts (c, a and b).
#[inline]
fn blocked_2d_applicable<D>(layouts: &[Layout<D>], size: usize) -> bool
where
    D: DimAPI,
{
    size >= TILE_SWITCH
        && layouts[0].ndim() == 2
        && layouts.iter().all(|l| l.shape().as_ref().iter().zip(l.stride().as_ref()).all(|(&d, &s)| !(d > 1 && s == 0)))
}

/// Blocked 2-D elementwise iteration (kernel body shared by the mutc/muta
/// drivers; the closure sees `(output_elem, input_elems...)` per element).
///
/// # Offsets and negative strides
///
/// All offsets are accumulated in `isize` (flipped views contribute negative
/// strides here) and cast to `usize` only after the full offset sum; a
/// `debug_assert` validates every element offset against the buffer length
/// in debug builds.
///
/// # Visit order
///
/// Elements are visited in tiled order rather than strict layout order.
/// Results are bit-identical for stateless per-element ops (all operators
/// rstsr ships): each output element is written exactly once from exactly
/// one input tuple, with no cross-element dependency. A *stateful* user
/// closure would observe the reordered visit sequence (the parallel path was
/// already order-free).
macro_rules! blocked_2d_iter {
    // emits the tile loops; $f is invoked as $f(&mut c[off_c], &a[off_a], ...)
    // inside the element loop; `c`, `a`, `b` are expressions evaluating to
    // the buffers; all offsets are isize, cast at the indexing site.
    ($c:expr, $lc:expr, $a:expr, $la:expr, $b:expr, $lb:expr, $f:expr) => {{
        let shape = $lc.shape().as_ref();
        let (dim0, dim1) = (shape[0], shape[1]);
        let sc = $lc.stride().as_ref();
        let sa = $la.stride().as_ref();
        let sb = $lb.stride().as_ref();
        // choose c's fastest axis as the element-innermost loop
        let (fast, slow) = if sc[0].abs() <= sc[1].abs() { (0usize, 1usize) } else { (1, 0) };
        let (dim_fast, dim_slow) = if fast == 0 { (dim0, dim1) } else { (dim1, dim0) };
        let (sc_fast, sc_slow) = (sc[fast], sc[slow]);
        let (sa_fast, sa_slow) = (sa[fast], sa[slow]);
        let (sb_fast, sb_slow) = (sb[fast], sb[slow]);
        // offset bases in isize; usize casts happen only at the indexing site
        let (oc, oa, ob) = ($lc.offset() as isize, $la.offset() as isize, $lb.offset() as isize);
        // tile grid: row-major in (slow, fast) so consecutive tiles are
        // adjacent along the output's fast axis
        for t_slow in (0..dim_slow).step_by(TILE) {
            let slow_end = (t_slow + TILE).min(dim_slow);
            for t_fast in (0..dim_fast).step_by(TILE) {
                let fast_end = (t_fast + TILE).min(dim_fast);
                for s in t_slow..slow_end {
                    let mut off_c = oc + (s as isize) * sc_slow + (t_fast as isize) * sc_fast;
                    let mut off_a = oa + (s as isize) * sa_slow + (t_fast as isize) * sa_fast;
                    let mut off_b = ob + (s as isize) * sb_slow + (t_fast as isize) * sb_fast;
                    for _ in t_fast..fast_end {
                        debug_assert!(
                            off_c >= 0 && (off_c as usize) < $c.len(),
                            "blocked 2-D iter: c offset out of bounds"
                        );
                        debug_assert!(
                            off_a >= 0 && (off_a as usize) < $a.len(),
                            "blocked 2-D iter: a offset out of bounds"
                        );
                        debug_assert!(
                            off_b >= 0 && (off_b as usize) < $b.len(),
                            "blocked 2-D iter: b offset out of bounds"
                        );
                        $f(&mut $c[off_c as usize], &$a[off_a as usize], &$b[off_b as usize]);
                        off_c += sc_fast;
                        off_a += sa_fast;
                        off_b += sb_fast;
                    }
                }
            }
        }
    }};
}

/// Whether the blocked 2-D path applies for a 2-operand (in-place) kernel:
/// same contract as [`blocked_2d_applicable`] (the slice holds the output
/// and input layouts; all of them are guard-checked).
#[inline]
fn blocked_2d_applicable_2<D>(layouts: &[Layout<D>], size: usize) -> bool
where
    D: DimAPI,
{
    blocked_2d_applicable(layouts, size)
}

macro_rules! blocked_2d_iter_2 {
    // 2-layout variant: output and one input (in-place ops). $f is invoked as
    // $f(&mut a[off_a], &b[off_b]).
    ($a:expr, $la:expr, $b:expr, $lb:expr, $f:expr) => {{
        let shape = $la.shape().as_ref();
        let (dim0, dim1) = (shape[0], shape[1]);
        let sa = $la.stride().as_ref();
        let sb = $lb.stride().as_ref();
        let (fast, slow) = if sa[0].abs() <= sa[1].abs() { (0usize, 1usize) } else { (1, 0) };
        let (dim_fast, dim_slow) = if fast == 0 { (dim0, dim1) } else { (dim1, dim0) };
        let (sa_fast, sa_slow) = (sa[fast], sa[slow]);
        let (sb_fast, sb_slow) = (sb[fast], sb[slow]);
        let (oa, ob) = ($la.offset() as isize, $lb.offset() as isize);
        for t_slow in (0..dim_slow).step_by(TILE) {
            let slow_end = (t_slow + TILE).min(dim_slow);
            for t_fast in (0..dim_fast).step_by(TILE) {
                let fast_end = (t_fast + TILE).min(dim_fast);
                for s in t_slow..slow_end {
                    let mut off_a = oa + (s as isize) * sa_slow + (t_fast as isize) * sa_fast;
                    let mut off_b = ob + (s as isize) * sb_slow + (t_fast as isize) * sb_fast;
                    for _ in t_fast..fast_end {
                        debug_assert!(
                            off_a >= 0 && (off_a as usize) < $a.len(),
                            "blocked 2-D iter: a offset out of bounds"
                        );
                        debug_assert!(
                            off_b >= 0 && (off_b as usize) < $b.len(),
                            "blocked 2-D iter: b offset out of bounds"
                        );
                        $f(&mut $a[off_a as usize], &$b[off_b as usize]);
                        off_a += sa_fast;
                        off_b += sb_fast;
                    }
                }
            }
        }
    }};
}

/// 4-layout (output `d` + three inputs) variant of [`blocked_2d_iter`]: same
/// contracts; the closure sees `(output_elem, input_elems...)` per element.
macro_rules! blocked_2d_iter_4 {
    ($d:expr, $ld:expr, $a:expr, $la:expr, $b:expr, $lb:expr, $c:expr, $lc:expr, $f:expr) => {{
        let shape = $ld.shape().as_ref();
        let (dim0, dim1) = (shape[0], shape[1]);
        let sd = $ld.stride().as_ref();
        let sa = $la.stride().as_ref();
        let sb = $lb.stride().as_ref();
        let sc = $lc.stride().as_ref();
        // choose d's fastest axis as the element-innermost loop
        let (fast, slow) = if sd[0].abs() <= sd[1].abs() { (0usize, 1usize) } else { (1, 0) };
        let (dim_fast, dim_slow) = if fast == 0 { (dim0, dim1) } else { (dim1, dim0) };
        let (sd_fast, sd_slow) = (sd[fast], sd[slow]);
        let (sa_fast, sa_slow) = (sa[fast], sa[slow]);
        let (sb_fast, sb_slow) = (sb[fast], sb[slow]);
        let (sc_fast, sc_slow) = (sc[fast], sc[slow]);
        // offset bases in isize; usize casts happen only at the indexing site
        let (od, oa, ob, oc) =
            ($ld.offset() as isize, $la.offset() as isize, $lb.offset() as isize, $lc.offset() as isize);
        // tile grid: row-major in (slow, fast) so consecutive tiles are
        // adjacent along the output's fast axis
        for t_slow in (0..dim_slow).step_by(TILE) {
            let slow_end = (t_slow + TILE).min(dim_slow);
            for t_fast in (0..dim_fast).step_by(TILE) {
                let fast_end = (t_fast + TILE).min(dim_fast);
                for s in t_slow..slow_end {
                    let mut off_d = od + (s as isize) * sd_slow + (t_fast as isize) * sd_fast;
                    let mut off_a = oa + (s as isize) * sa_slow + (t_fast as isize) * sa_fast;
                    let mut off_b = ob + (s as isize) * sb_slow + (t_fast as isize) * sb_fast;
                    let mut off_c = oc + (s as isize) * sc_slow + (t_fast as isize) * sc_fast;
                    for _ in t_fast..fast_end {
                        debug_assert!(
                            off_d >= 0 && (off_d as usize) < $d.len(),
                            "blocked 2-D iter: d offset out of bounds"
                        );
                        debug_assert!(
                            off_a >= 0 && (off_a as usize) < $a.len(),
                            "blocked 2-D iter: a offset out of bounds"
                        );
                        debug_assert!(
                            off_b >= 0 && (off_b as usize) < $b.len(),
                            "blocked 2-D iter: b offset out of bounds"
                        );
                        debug_assert!(
                            off_c >= 0 && (off_c as usize) < $c.len(),
                            "blocked 2-D iter: c offset out of bounds"
                        );
                        $f(&mut $d[off_d as usize], &$a[off_a as usize], &$b[off_b as usize], &$c[off_c as usize]);
                        off_d += sd_fast;
                        off_a += sa_fast;
                        off_b += sb_fast;
                        off_c += sc_fast;
                    }
                }
            }
        }
    }};
}

pub fn op_mutc_refa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        let lb = &layouts_contig[2];
        layout_col_major_dim_dispatch_3(lc, la, lb, |(idx_c, idx_a, idx_b)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b[idx_b + i]);
            }
        })
    } else if blocked_2d_applicable(&layouts_full, layouts_full[0].size()) {
        // blocked 2-D iteration for fully strided problems (e.g. a + b.t()):
        // see `blocked_2d_iter!` for the visit-order and offset contracts
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        blocked_2d_iter!(c, lc, a, la, b, lb, f);
        Ok(())
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        layout_col_major_dim_dispatch_3(lc, la, lb, |(idx_c, idx_a, idx_b)| {
            f(&mut c[idx_c], &a[idx_a], &b[idx_b]);
        })
    }
}

/// 3-input (output `d` + three refs) elementwise kernel, e.g. `where(cond, x, y)`.
pub fn op_mutd_refa_refb_refc_func_cpu_serial<TA, TB, TC, TD, D>(
    d: &mut [MaybeUninit<TD>],
    ld: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    c: &[TC],
    lc: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TD>, &TA, &TB, &TC),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[ld, la, lb, lc], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let ld = &layouts_contig[0];
        let la = &layouts_contig[1];
        let lb = &layouts_contig[2];
        let lc = &layouts_contig[3];
        layout_col_major_dim_dispatch_4(ld, la, lb, lc, |(idx_d, idx_a, idx_b, idx_c)| {
            for i in 0..size_contig {
                f(&mut d[idx_d + i], &a[idx_a + i], &b[idx_b + i], &c[idx_c + i]);
            }
        })
    } else if blocked_2d_applicable(&layouts_full, layouts_full[0].size()) {
        // blocked 2-D iteration for fully strided problems; see
        // `blocked_2d_iter_4!` for the visit-order and offset contracts
        let ld = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        let lc = &layouts_full[3];
        blocked_2d_iter_4!(d, ld, a, la, b, lb, c, lc, f);
        Ok(())
    } else {
        let ld = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        let lc = &layouts_full[3];
        layout_col_major_dim_dispatch_4(ld, la, lb, lc, |(idx_d, idx_a, idx_b, idx_c)| {
            f(&mut d[idx_d], &a[idx_a], &b[idx_b], &c[idx_c]);
        })
    }
}
pub fn op_mutc_refa_numb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a[idx_a + i], &b);
            }
        })
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            f(&mut c[idx_c], &a[idx_a], &b);
        })
    }
}

pub fn op_mutc_numa_refb_func_cpu_serial<TA, TB, TC, D>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TC>, &TA, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let lc = &layouts_contig[0];
        let lb = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, lb, |(idx_c, idx_b)| {
            for i in 0..size_contig {
                f(&mut c[idx_c + i], &a, &b[idx_b + i]);
            }
        })
    } else {
        let lc = &layouts_full[0];
        let lb = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, lb, |(idx_c, idx_b)| {
            f(&mut c[idx_c], &a, &b[idx_b]);
        })
    }
}

pub fn op_muta_refb_func_cpu_serial<TA, TB, D>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<TA>, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // contiguous iteration if possible, otherwise use iterator of layout
    if size_contig >= CONTIG_SWITCH {
        let la = &layouts_contig[0];
        let lb = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(la, lb, |(idx_a, idx_b)| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b[idx_b + i]);
            }
        })
    } else if blocked_2d_applicable_2(&layouts_full, layouts_full[0].size()) {
        // blocked 2-D iteration for fully strided in-place problems
        // (e.g. c += b.t()): see `blocked_2d_iter_2!` for the contracts
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        blocked_2d_iter_2!(a, la, b, lb, f);
        Ok(())
    } else {
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        layout_col_major_dim_dispatch_2(la, lb, |(idx_a, idx_b)| {
            f(&mut a[idx_a], &b[idx_b]);
        })
    }
}

pub fn op_muta_numb_func_cpu_serial<TA, TB, D>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: TB,
    mut f: impl FnMut(&mut MaybeUninit<TA>, &TB),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i], &b);
            }
        })
    } else {
        let la = &layout;
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            f(&mut a[idx_a], &b);
        })
    }
}

pub fn op_muta_func_cpu_serial<T, D>(
    a: &mut [MaybeUninit<T>],
    la: &Layout<D>,
    mut f: impl FnMut(&mut MaybeUninit<T>),
) -> Result<()>
where
    D: DimAPI,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            for i in 0..size_contig {
                f(&mut a[idx_a + i]);
            }
        })
    } else {
        let la = &layout;
        layout_col_major_dim_dispatch_1(la, |idx_a| {
            f(&mut a[idx_a]);
        })
    }
}
