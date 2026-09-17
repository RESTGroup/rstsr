//! Naive implementation of matrix transpose

use crate::prelude_dev::*;

const BLOCK_SIZE: usize = 64;

/// Change order (row/col-major) a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (row-major) to `c` (col-major).
/// If shape or stride is not compatible, an error will be returned.
///
/// This function does not take thread-pool as argument, so the caller should
/// take care of thread-pool management.
pub fn orderchange_out_r2c_ix2_cpu_rayon_no_pool<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < 16 * BLOCK_SIZE * BLOCK_SIZE {
        return orderchange_out_r2c_ix2_cpu_serial(c, lc, a, la);
    }

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout, "This function requires shape identity")?;
    rstsr_assert_eq!(sc[1], sa[1], InvalidLayout, "This function requires shape identity")?;
    let [nrow, ncol] = *sa;

    // stride check
    rstsr_assert_eq!(lc.stride()[0], 1, InvalidLayout, "This function requires col-major output")?;
    rstsr_assert_eq!(la.stride()[1], 1, InvalidLayout, "This function requires row-major input")?;

    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let lda = la.stride()[0];
    let ldc = lc.stride()[1];

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    (0..ncol).into_par_iter().step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        let (j_start, j_end) = (j_start as isize, j_end as isize);
        (0..nrow).into_par_iter().step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            let (i_start, i_end) = (i_start as isize, i_end as isize);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i * lda + j) as usize;
                    let dst_idx = (offset_c + j * ldc + i) as usize;

                    unsafe {
                        // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                        // (relaxed load; `c` is never reassigned through it). Each (i, j) pair
                        // maps to a unique `dst_idx` — disjoint writes across parallel tasks.
                        let c_ptr = thr_c.load(Ordering::Relaxed).add(dst_idx);
                        *c_ptr = a[src_idx].clone();
                    }
                }
            }
        });
    });

    Ok(())
}

/// Change order (row/col-major) a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (row-major) to `c` (col-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn orderchange_out_r2c_ix2_cpu_rayon<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < 16 * BLOCK_SIZE * BLOCK_SIZE || pool.is_none() {
        return orderchange_out_r2c_ix2_cpu_serial(c, lc, a, la);
    }

    pool.unwrap().install(|| orderchange_out_r2c_ix2_cpu_rayon_no_pool(c, lc, a, la))
}

/// Change order (row/col-major) a matrix out-place, promote/uninit variants
/// (rayon twins of the `*_promote_cpu_serial` kernels in `cpu_serial`).
///
/// Same blocked traversal and stride guards as the serial kernels; writes go
/// through raw pointers guarded by `debug_assert!` bounds contracts (the
/// per-element bounds check is redundant here: the layout guard plus the
/// full (i, j) write-once coverage guarantee in-bounds access). The uninit
/// variant is sound because every output element is written exactly once.
#[duplicate_item(
    func_name
        func_serial TypeC TypeA func_write
    ;
    [orderchange_out_r2c_ix2_promote_cpu_rayon]
        [orderchange_out_r2c_ix2_promote_cpu_serial]
        [TC] [TA]
        [(*c_ptr.add(dst_idx)) = a[src_idx].clone().into_cast()]
    ;
    [orderchange_out_r2c_ix2_uninit_promote_cpu_rayon]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA]
        [(*c_ptr.add(dst_idx)).write(a[src_idx].clone().into_cast())]
    ;
)]
pub fn func_name<TC, TA>(
    c: &mut [TypeC],
    lc: &Layout<Ix2>,
    a: &[TypeA],
    la: &Layout<Ix2>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TC: Clone + Send + Sync,
    TA: Clone + Send + Sync + DTypeCastAPI<TC>,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < 16 * BLOCK_SIZE * BLOCK_SIZE || pool.is_none() {
        return func_serial(c, lc, a, la);
    }

    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout, "This function requires shape identity")?;
    rstsr_assert_eq!(sc[1], sa[1], InvalidLayout, "This function requires shape identity")?;
    let [nrow, ncol] = *sa;

    // stride check
    rstsr_assert_eq!(lc.stride()[0], 1, InvalidLayout, "This function requires col-major output")?;
    rstsr_assert_eq!(la.stride()[1], 1, InvalidLayout, "This function requires row-major input")?;

    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let lda = la.stride()[0];
    let ldc = lc.stride()[1];

    // pass mutable reference in parallel region
    let thr_c = AtomicPtr::new(c.as_mut_ptr());
    (0..ncol).into_par_iter().step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        let (j_start, j_end) = (j_start as isize, j_end as isize);
        (0..nrow).into_par_iter().step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            let (i_start, i_end) = (i_start as isize, i_end as isize);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i * lda + j) as usize;
                    let dst_idx = (offset_c + j * ldc + i) as usize;
                    debug_assert!(src_idx < a.len() && dst_idx < c.len());
                    unsafe {
                        // SAFETY: `c_ptr` is `c`'s base pointer hoisted through `AtomicPtr`
                        // (relaxed load; `c` is never reassigned through it). Each (i, j) write
                        // touches a unique layout position of `c` — disjoint across tasks.
                        let c_ptr = thr_c.load(Ordering::Relaxed);
                        func_write;
                    }
                }
            }
        });
    });

    Ok(())
}

/// Change order (row/col-major) a matrix out-place, promote/uninit c2r
/// wrappers with pool management (see the r2c rayon kernels above).
#[duplicate_item(
    func_name
        TypeC TypeA func_r2c
    ;
    [orderchange_out_c2r_ix2_promote_cpu_rayon]
        [TC] [TA]
        [orderchange_out_r2c_ix2_promote_cpu_rayon]
    ;
    [orderchange_out_c2r_ix2_uninit_promote_cpu_rayon]
        [MaybeUninit<TC>] [TA]
        [orderchange_out_r2c_ix2_uninit_promote_cpu_rayon]
    ;
)]
pub fn func_name<TC, TA>(
    c: &mut [TypeC],
    lc: &Layout<Ix2>,
    a: &[TypeA],
    la: &Layout<Ix2>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TC: Clone + Send + Sync,
    TA: Clone + Send + Sync + DTypeCastAPI<TC>,
{
    let lc = lc.reverse_axes();
    let la = la.reverse_axes();
    func_r2c(c, &lc, a, &la, pool)
}

/// Change order (row/col-major) a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (col-major) to `c` (row-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn orderchange_out_c2r_ix2_cpu_rayon<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    let lc = lc.reverse_axes();
    let la = la.reverse_axes();
    orderchange_out_r2c_ix2_cpu_rayon(c, &lc, a, &la, pool)
}
