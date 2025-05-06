//! Naive implementation of matrix transpose

use crate::prelude_dev::*;
use rayon::prelude::*;

const BLOCK_SIZE: usize = 64;

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

    pool.unwrap().install(|| {
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
                            let c_ptr = c.as_ptr().add(dst_idx) as *mut T;
                            *c_ptr = a[src_idx].clone();
                        }
                    }
                }
            });
        })
    });

    Ok(())
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
