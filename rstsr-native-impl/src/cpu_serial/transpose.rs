//! Naive implementation of matrix transpose

use crate::prelude_dev::*;

const BLOCK_SIZE: usize = 64;

/// Transpose a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (row-major) to `c` (col-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn transpose_out_r2c_ix2_cpu_serial<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
) -> Result<()>
where
    T: Clone,
{
    // shape check
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[1], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sa[0], InvalidLayout)?;
    let [nrow, ncol] = *sa;

    // stride check
    rstsr_assert_eq!(lc.stride()[0], 1, InvalidLayout)?;
    rstsr_assert_eq!(la.stride()[1], 1, InvalidLayout)?;

    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let lda = la.stride()[0];
    let ldc = lc.stride()[1];

    (0..ncol).step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        let (j_start, j_end) = (j_start as isize, j_end as isize);
        (0..nrow).step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            let (i_start, i_end) = (i_start as isize, i_end as isize);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i * lda + j) as usize;
                    let dst_idx = (offset_c + j * ldc + i) as usize;
                    c[dst_idx] = a[src_idx].clone();
                }
            }
        });
    });

    Ok(())
}

/// Transpose a matrix out-place using a naive algorithm.
///
/// Transpose from `a` (col-major) to `c` (row-major).
/// If shape or stride is not compatible, an error will be returned.
pub fn transpose_out_c2r_ix2_cpu_serial<T>(
    c: &mut [T],
    lc: &Layout<Ix2>,
    a: &[T],
    la: &Layout<Ix2>,
) -> Result<()>
where
    T: Clone,
{
    // shape check
    let lc = lc.reverse_axes();
    let la = la.reverse_axes();
    transpose_out_r2c_ix2_cpu_serial(c, &lc, a, &la)
}
