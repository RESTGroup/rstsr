//! Assembly of the complex outputs of the non-symmetric eigendecomposition
//! drivers (`?GEEV`, `?GGEV`) from the real-valued slots LAPACK uses for a
//! real matrix: the split spectrum `(wr, wi)` and the eigenvector matrices
//! packed by the conjugate-pair rule. The real vendor files call these after
//! the Fortran call; the complex ones pass the slots through untouched.

use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

/// Pack the split real spectrum `(wr, wi)` into the complex eigenvalues `w`.
pub fn pack_eigenvalues<R>(wr: &[R], wi: &[R], w: &mut [Complex<R>])
where
    R: ComplexFloat,
{
    assert_eq!(w.len(), wr.len());
    assert_eq!(w.len(), wi.len());
    for ((w, &re), &im) in w.iter_mut().zip(wr).zip(wi) {
        *w = Complex::new(re, im);
    }
}

/// Expand a conjugate-pair packed real eigenvector matrix into complex columns.
///
/// `v_real` holds the `(n, n)` real matrix in `order` with leading dimension
/// `ldv` (`n * ldv` elements), `wi` is the imaginary part of the spectrum, and
/// `out` receives the complex columns in the same order and leading dimension.
/// A real eigenvalue `j` takes the column `v_real[:, j]`; a conjugate pair
/// `(j, j+1)` (positive imaginary part first) gives
/// `v_real[:, j] ± i v_real[:, j+1]`. A non-zero imaginary part without its
/// partner (never emitted by a conforming LAPACK) keeps a real column.
pub fn unpack_real_eigvecs<R>(order: FlagOrder, n: usize, ldv: usize, v_real: &[R], wi: &[R], out: &mut [Complex<R>])
where
    R: ComplexFloat,
{
    assert!(n <= wi.len());
    let len = n * ldv;
    assert!(v_real.len() >= len);
    assert!(out.len() >= len);
    let (row_stride, col_stride) = match order {
        ColMajor => (1, ldv),
        RowMajor => (ldv, 1),
    };
    let mut j = 0;
    while j < n {
        // LAPACK emits pairs as (positive imaginary part, its conjugate)
        let paired = !wi[j].is_zero() && j + 1 < n && !wi[j + 1].is_zero();
        debug_assert!(paired || wi[j].is_zero(), "unpaired complex eigenvalue");
        if !paired {
            for i in 0..n {
                let k = i * row_stride + j * col_stride;
                out[k] = Complex::new(v_real[k], R::zero());
            }
            j += 1;
        } else {
            for i in 0..n {
                let k0 = i * row_stride + j * col_stride;
                let k1 = i * row_stride + (j + 1) * col_stride;
                let re = v_real[k0];
                let im = v_real[k1];
                out[k0] = Complex::new(re, im);
                out[k1] = Complex::new(re, -im);
            }
            j += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // LAPACK's packing: a real eigenvalue 2 and the pair 3 ± 4i, whose packed
    // columns are x and y with v(j) = x + i y and v(j+1) = x - i y
    #[test]
    fn pack_and_unpack_col_major() {
        let wr = [2.0, 3.0, 3.0];
        let wi = [0.0, 4.0, -4.0];
        let mut w = vec![Complex::new(0.0, 0.0); 3];
        pack_eigenvalues(&wr, &wi, &mut w);
        assert_eq!(w, vec![Complex::new(2.0, 0.0), Complex::new(3.0, 4.0), Complex::new(3.0, -4.0)]);

        // columns (ld = 3): [1, 0, 0], x = [1, 1, 0], y = [2, -1, 0]
        let v_real = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, -1.0, 0.0];
        let mut out = vec![Complex::new(0.0, 0.0); 9];
        unpack_real_eigvecs(ColMajor, 3, 3, &v_real, &wi, &mut out);
        assert_eq!(out[0], Complex::new(1.0, 0.0));
        assert_eq!(out[3], Complex::new(1.0, 2.0));
        assert_eq!(out[4], Complex::new(1.0, -1.0));
        assert_eq!(out[6], Complex::new(1.0, -2.0));
        assert_eq!(out[7], Complex::new(1.0, 1.0));
    }

    // the same matrix row-major (ld = 3): pair first, real column last
    #[test]
    fn unpack_row_major() {
        let wi = [4.0, -4.0, 0.0];
        let v_real = [1.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        let mut out = vec![Complex::new(0.0, 0.0); 9];
        unpack_real_eigvecs(RowMajor, 3, 3, &v_real, &wi, &mut out);
        assert_eq!(out[0], Complex::new(1.0, 2.0));
        assert_eq!(out[1], Complex::new(1.0, -2.0));
        assert_eq!(out[3], Complex::new(1.0, -1.0));
        assert_eq!(out[4], Complex::new(1.0, 1.0));
        assert_eq!(out[8], Complex::new(1.0, 0.0));
    }

    // a padded leading dimension skips the padding in both orders
    #[test]
    fn unpack_padded_ld() {
        let wi = [0.0, 4.0, -4.0];
        // column-major, ld = 4: column j starts at 4 j
        let v_real = [1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0, 9.0, 7.0, 8.0, 9.0, 9.0];
        let mut out = vec![Complex::new(0.0, 0.0); 12];
        unpack_real_eigvecs(ColMajor, 3, 4, &v_real, &wi, &mut out);
        assert_eq!(out[2], Complex::new(3.0, 0.0));
        assert_eq!(out[5], Complex::new(5.0, 8.0));
        assert_eq!(out[9], Complex::new(5.0, -8.0));
    }

    // a non-zero imaginary part without its partner stays a real column
    // (release builds read no column beyond `n`; debug builds assert instead)
    #[cfg(not(debug_assertions))]
    #[test]
    fn unpack_unpaired_keeps_real_column() {
        let wi = [0.0, 4.0];
        let v_real = [1.0, 2.0, 3.0, 4.0];
        let mut out = vec![Complex::new(0.0, 0.0); 4];
        unpack_real_eigvecs(ColMajor, 2, 2, &v_real, &wi, &mut out);
        assert_eq!(out, vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0)
        ]);
    }
}
