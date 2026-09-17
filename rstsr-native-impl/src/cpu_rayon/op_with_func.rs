use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Since current task is not intensive to each element, this value is large.
const PARALLEL_SWITCH: usize = 4096;

// Minimum problem size for the blocked 2-D iteration path (mirrors the
// serial file); smaller problems fall through to the serial kernel.
const TILE_SWITCH: usize = 4096;
// Tile edge (elements); within a [TILE, TILE] tile each layout's inner-axis
// hop stays within TILE cache lines for the whole tile pass.
const TILE: usize = 64;

/// Guard for the blocked 2-D path: 2-D, large enough, and **no layout has a
/// broadcast axis (shape > 1 with stride 0)** on any axis — broadcast
/// operands keep the generic iterator path. All participating layouts are
/// guard-checked.
///
/// Performance caveat: the blocked path parallelizes over slow-axis tile
/// bands only (`ceil(dim_slow / TILE)` tasks); very tall-skinny shapes
/// (small `dim_slow`) may see less parallelism here than the generic
/// outer-parallel path (e.g. 70×10000 measured slower on 8 threads).
#[inline]
fn blocked_2d_applicable_rayon<D>(layouts: &[Layout<D>], size: usize) -> bool
where
    D: DimAPI,
{
    size >= TILE_SWITCH
        && layouts[0].ndim() == 2
        && layouts.iter().all(|l| l.shape().as_ref().iter().zip(l.stride().as_ref()).all(|(&d, &s)| !(d > 1 && s == 0)))
}

/// Read-only view of the blocked-2D iteration geometry (axis roles, tile
/// grid, stride sums) shared by the rayon tile kernels. Offsets are isize;
/// usize casts happen only at the addressing site.
struct Blocked2DGeom {
    dim_fast: usize,
    dim_slow: usize,
    sc_fast: isize,
    sc_slow: isize,
    sa_fast: isize,
    sa_slow: isize,
    sb_fast: isize,
    sb_slow: isize,
    oc: isize,
    oa: isize,
    ob: isize,
    n_tiles_fast: usize,
    n_tiles_slow: usize,
}

#[inline]
fn blocked_2d_geom<D>(lc: &Layout<D>, la: &Layout<D>, lb: &Layout<D>) -> Blocked2DGeom
where
    D: DimAPI,
{
    let shape = lc.shape().as_ref();
    let (dim0, dim1) = (shape[0], shape[1]);
    let sc = lc.stride().as_ref();
    let sa = la.stride().as_ref();
    let sb = lb.stride().as_ref();
    // choose c's fastest axis as the element-innermost loop
    let (fast, slow) = if sc[0].abs() <= sc[1].abs() { (0usize, 1usize) } else { (1, 0) };
    let (dim_fast, dim_slow) = if fast == 0 { (dim0, dim1) } else { (dim1, dim0) };
    Blocked2DGeom {
        dim_fast,
        dim_slow,
        sc_fast: sc[fast],
        sc_slow: sc[slow],
        sa_fast: sa[fast],
        sa_slow: sa[slow],
        sb_fast: sb[fast],
        sb_slow: sb[slow],
        oc: lc.offset() as isize,
        oa: la.offset() as isize,
        ob: lb.offset() as isize,
        n_tiles_fast: dim_fast.div_ceil(TILE),
        n_tiles_slow: dim_slow.div_ceil(TILE),
    }
}

/// Blocked 2-D elementwise iteration, parallel over tile-row bands (each
/// rayon task covers a [TILE, dim_fast] band of the output).
///
/// Visit-order note: elements are visited in tiled order; bit-identical for
/// stateless per-element ops, and this path (like the whole rayon kernel)
/// was already order-free. Aliasing/bounds contract matches the other
/// raw-pointer paths in this file: disjoint buffers, layouts consistent with
/// shapes (offsets are debug-asserted).
#[allow(clippy::too_many_arguments)]
fn blocked_2d_3layouts_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    D: DimAPI,
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    let g = blocked_2d_geom(lc, la, lb);
    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    let task = || {
        (0..g.n_tiles_slow).into_par_iter().for_each(|t_slow| {
            let slow_begin = t_slow * TILE;
            let slow_end = (slow_begin + TILE).min(g.dim_slow);
            // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr` (relaxed
            // load; `c` is never reassigned through it). Each parallel task writes the
            // disjoint region of `c` given by its own tile band (offsets from distinct
            // output-layout positions; bounds debug-asserted in the loop).
            let c_ptr = thr_c.load(Ordering::Relaxed);
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            for t_fast in 0..g.n_tiles_fast {
                let fast_begin = t_fast * TILE;
                let fast_end = (fast_begin + TILE).min(g.dim_fast);
                for s in slow_begin..slow_end {
                    let mut off_c = g.oc + (s as isize) * g.sc_slow + (fast_begin as isize) * g.sc_fast;
                    let mut off_a = g.oa + (s as isize) * g.sa_slow + (fast_begin as isize) * g.sa_fast;
                    let mut off_b = g.ob + (s as isize) * g.sb_slow + (fast_begin as isize) * g.sb_fast;
                    for _ in fast_begin..fast_end {
                        debug_assert!(
                            off_c >= 0 && (off_c as usize) < c.len(),
                            "blocked 2-D iter: c offset out of bounds"
                        );
                        debug_assert!(
                            off_a >= 0 && (off_a as usize) < a.len(),
                            "blocked 2-D iter: a offset out of bounds"
                        );
                        debug_assert!(
                            off_b >= 0 && (off_b as usize) < b.len(),
                            "blocked 2-D iter: b offset out of bounds"
                        );
                        unsafe {
                            f(&mut *c_ptr.add(off_c as usize), &*a_ptr.add(off_a as usize), &*b_ptr.add(off_b as usize))
                        };
                        off_c += g.sc_fast;
                        off_a += g.sa_fast;
                        off_b += g.sb_fast;
                    }
                }
            }
        });
    };
    pool.map_or_else(task, |pool| pool.install(task));
    Ok(())
}

/// 2-layout (in-place) variant of [`blocked_2d_3layouts_cpu_rayon`]; same
/// contract and visit-order note.
fn blocked_2d_2layouts_cpu_rayon<TA, TB, D, F>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    D: DimAPI,
    TA: Send + Sync,
    TB: Send + Sync,
    F: Fn(&mut MaybeUninit<TA>, &TB) + ?Sized + Send + Sync,
{
    let shape = la.shape().as_ref();
    let (dim0, dim1) = (shape[0], shape[1]);
    let sa = la.stride().as_ref();
    let sb = lb.stride().as_ref();
    let (fast, slow) = if sa[0].abs() <= sa[1].abs() { (0usize, 1usize) } else { (1, 0) };
    let (dim_fast, dim_slow) = if fast == 0 { (dim0, dim1) } else { (dim1, dim0) };
    let (sa_fast, sa_slow) = (sa[fast], sa[slow]);
    let (sb_fast, sb_slow) = (sb[fast], sb[slow]);
    let (oa, ob) = (la.offset() as isize, lb.offset() as isize);
    let n_tiles_fast = dim_fast.div_ceil(TILE);
    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());
    let task = || {
        (0..dim_slow.div_ceil(TILE)).into_par_iter().for_each(|t_slow| {
            let slow_begin = t_slow * TILE;
            let slow_end = (slow_begin + TILE).min(dim_slow);
            // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr` (relaxed
            // load; `a` is never reassigned through it). Each parallel task writes the
            // disjoint region of `a` given by its own tile band (offsets from distinct
            // output-layout positions; bounds debug-asserted in the loop).
            let a_ptr = thr_a.load(Ordering::Relaxed);
            let b_ptr = b.as_ptr();
            for t_fast in 0..n_tiles_fast {
                let fast_begin = t_fast * TILE;
                let fast_end = (fast_begin + TILE).min(dim_fast);
                for s in slow_begin..slow_end {
                    let mut off_a = oa + (s as isize) * sa_slow + (fast_begin as isize) * sa_fast;
                    let mut off_b = ob + (s as isize) * sb_slow + (fast_begin as isize) * sb_fast;
                    for _ in fast_begin..fast_end {
                        debug_assert!(
                            off_a >= 0 && (off_a as usize) < a.len(),
                            "blocked 2-D iter: a offset out of bounds"
                        );
                        debug_assert!(
                            off_b >= 0 && (off_b as usize) < b.len(),
                            "blocked 2-D iter: b offset out of bounds"
                        );
                        unsafe { f(&mut *a_ptr.add(off_a as usize), &*b_ptr.add(off_b as usize)) };
                        off_a += sa_fast;
                        off_b += sb_fast;
                    }
                }
            }
        });
    };
    pool.map_or_else(task, |pool| pool.install(task));
    Ok(())
}

/* #region op_func definition */

#[allow(clippy::too_many_arguments)]
pub fn op_mutc_refa_refb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Sync + Send,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_refa_refb_func_cpu_serial(c, lc, a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let la = &layouts_outer[1];
        let lb = &layouts_outer[2];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_a, idx_b)| unsafe {
                // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `c` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_c` of the validated output layout.
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_a, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `c` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let c_ptr = thr_c.load(Ordering::Relaxed);
                    f(&mut *c_ptr.add(idx_c + idx), &a[idx_a + idx], &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else if blocked_2d_applicable_rayon(&layouts_full, size) {
        // blocked 2-D iteration for fully strided problems (e.g. a + b.t())
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        blocked_2d_3layouts_cpu_rayon(c, lc, a, la, b, lb, f, pool)
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let lb = &layouts_full[2];
        let func = |(idx_c, idx_a, idx_b)| unsafe {
            // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `c` is never reassigned through it). Each task writes
            // the disjoint element at `idx_c` of the validated output layout.
            let c_ptr = thr_c.load(Ordering::Relaxed);
            f(&mut *c_ptr.add(idx_c), &a[idx_a], &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_3(lc, la, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_mutc_refa_numb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: &[TA],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_refa_numb_func_cpu_serial(c, lc, a, la, b, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let la = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_a)| unsafe {
                // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `c` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_c` of the validated output layout.
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a[idx_a + idx], &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_a)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `c` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let c_ptr = thr_c.load(Ordering::Relaxed);
                    f(&mut *c_ptr.add(idx_c + idx), &a[idx_a + idx], &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let func = |(idx_c, idx_a)| unsafe {
            // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `c` is never reassigned through it). Each task writes
            // the disjoint element at `idx_c` of the validated output layout.
            let c_ptr = thr_c.load(Ordering::Relaxed);
            f(&mut *c_ptr.add(idx_c), &a[idx_a], &b);
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_mutc_numa_refb_func_cpu_rayon<TA, TB, TC, D, F>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<D>,
    a: TA,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    TC: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TC>, &TA, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_mutc_numa_refb_func_cpu_serial(c, lc, a, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let lc = &layouts_outer[0];
        let lb = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_c, idx_b)| unsafe {
                // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `c` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_c` of the validated output layout.
                let c_ptr = thr_c.load(Ordering::Relaxed).add(idx_c);
                (0..size_contig).for_each(|idx| {
                    f(&mut *c_ptr.add(idx), &a, &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_c, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `c` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let c_ptr = thr_c.load(Ordering::Relaxed);
                    f(&mut *c_ptr.add(idx_c + idx), &a, &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let lb = &layouts_full[1];
        let func = |(idx_c, idx_b)| unsafe {
            // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `c` is never reassigned through it). Each task writes
            // the disjoint element at `idx_c` of the validated output layout.
            let c_ptr = thr_c.load(Ordering::Relaxed);
            f(&mut *c_ptr.add(idx_c), &a, &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_refb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: &[TB],
    lb: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TA>, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_refb_func_cpu_serial(a, la, b, lb, f);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[la, lb], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_outer, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let la = &layouts_outer[0];
        let lb = &layouts_outer[1];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |(idx_a, idx_b)| unsafe {
                // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `a` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_a` of the validated output layout.
                let a_ptr = thr_a.load(Ordering::Relaxed).add(idx_a);
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx), &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |(idx_a, idx_b)| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `a` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let a_ptr = thr_a.load(Ordering::Relaxed);
                    f(&mut *a_ptr.add(idx_a + idx), &b[idx_b + idx]);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else if blocked_2d_applicable_rayon(&layouts_full, la.size()) {
        // blocked 2-D iteration for fully strided in-place problems
        // (e.g. c += b.t())
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        blocked_2d_2layouts_cpu_rayon(a, la, b, lb, f, pool)
    } else {
        // not possible for contiguous assign
        let la = &layouts_full[0];
        let lb = &layouts_full[1];
        let func = |(idx_a, idx_b): (usize, usize)| unsafe {
            // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `a` is never reassigned through it). Each task writes
            // the disjoint element at `idx_a` of the validated output layout.
            let a_ptr = thr_a.load(Ordering::Relaxed);
            f(&mut *a_ptr.add(idx_a), &b[idx_b]);
        };
        let task = || layout_col_major_dim_dispatch_par_2(la, lb, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_numb_func_cpu_rayon<TA, TB, D, F>(
    a: &mut [MaybeUninit<TA>],
    la: &Layout<D>,
    b: TB,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TA: Send + Sync,
    TB: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<TA>, &TB) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_numb_func_cpu_serial(a, la, b, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        // parallel for outer iteration
        let la = &layout_contig[0];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |idx_a| unsafe {
                // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `a` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_a` of the validated output layout.
                let a_ptr = thr_a.load(Ordering::Relaxed).add(idx_a);
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx), &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |idx_a| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `a` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let a_ptr = thr_a.load(Ordering::Relaxed);
                    f(&mut *a_ptr.add(idx_a + idx), &b);
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        // not possible for contiguous assign
        let func = |idx_a| unsafe {
            // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `a` is never reassigned through it). Each task writes
            // the disjoint element at `idx_a` of the validated output layout.
            let a_ptr = thr_a.load(Ordering::Relaxed);
            f(&mut *a_ptr.add(idx_a), &b);
        };
        let task = || layout_col_major_dim_dispatch_par_1(&layout, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn op_muta_func_cpu_rayon<T, D, F>(
    a: &mut [MaybeUninit<T>],
    la: &Layout<D>,
    f: &mut F,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Send + Sync,
    D: DimAPI,
    F: Fn(&mut MaybeUninit<T>) + ?Sized + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = la.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return op_muta_func_cpu_serial(a, la, f);
    }

    // re-align layouts
    let layout = translate_to_col_major_unary(la, TensorIterOrder::G)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());

    // actual parallel iteration
    if size_contig >= CONTIG_SWITCH {
        let la = &layout_contig[0];
        if size_contig < PARALLEL_SWITCH {
            // not parallel inner iteration
            let func = |idx_a| unsafe {
                // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                // (relaxed load; `a` is never reassigned through it). Each task writes
                // the disjoint contiguous run at `idx_a` of the validated output layout.
                let a_ptr = thr_a.load(Ordering::Relaxed).add(idx_a);
                (0..size_contig).for_each(|idx| {
                    f(&mut *a_ptr.add(idx));
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            // parallel inner iteration
            let func = |idx_a| unsafe {
                (0..size_contig).into_par_iter().for_each(|idx| {
                    // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
                    // (relaxed load; `a` is never reassigned through it). Each task writes
                    // the single element at a disjoint offset of the validated output layout.
                    let a_ptr = thr_a.load(Ordering::Relaxed);
                    f(&mut *a_ptr.add(idx_a + idx));
                });
            };
            let task = || layout_col_major_dim_dispatch_par_1(la, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        let func = |idx_a| unsafe {
            // SAFETY: `a_ptr` is `a`'s base pointer hoisted through `AtomicPtr`
            // (relaxed load; `a` is never reassigned through it). Each task writes
            // the disjoint element at `idx_a` of the validated output layout.
            let a_ptr = thr_a.load(Ordering::Relaxed);
            f(&mut *a_ptr.add(idx_a));
        };
        let task = || layout_col_major_dim_dispatch_par_1(&layout, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}
