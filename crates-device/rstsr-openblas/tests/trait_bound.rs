//! Trait-bundle sufficiency test: generic code bounded only by
//! `LapackDriverAPI<T>` must be able to call the core matmul entry points
//! (including the `MaybeUninit` raw projection that `matmul_from` needs),
//! and the members absorbed into `BlasDriverBaseAPI` from the linalg tier:
//! the complex counterpart of a real `T`, vector (Ix1) assignment, the
//! index dtypes, in-place mutation, and the n-dim norm ops.
//!
//! Regression for the `261015/lapackdriverapi-trait-bound` change and the
//! `261128/blas-base-trait-bounds` change.

// device alias for the shared test style (rstsr-blas-traits/src/blas_tests_impl)
pub use rstsr_openblas::DeviceOpenBLAS as DeviceBLAS;

use num::{Complex, FromPrimitive, One};
use rstsr::prelude::rt::blas::{BlasFloat, LapackDriverAPI};
use rstsr::prelude::rt::Result;
use rstsr::prelude::*;
use rstsr_core::prelude_dev::{op_muta_func, MaybeUninit};

fn generic_matmul<B, T>(a: &Tensor<T, B, Ix2>, b: &Tensor<T, B, Ix2>) -> Tensor<T, B, Ix2>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    rt::matmul(a, b)
}

fn generic_matmul_scratch<B, T>(mut scratch: TensorMut<T, B, Ix2>, a: TensorView<T, B, Ix2>, b: TensorView<T, B, Ix2>)
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    scratch.matmul_from(a, b, T::one(), T::zero());
}

fn generic_matmul_dyn<B, T>(scratch_vec: &mut [T], a: &Tensor<T, B, IxD>, b: &Tensor<T, B, Ix2>, device: &B)
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let shape = [a.shape()[0], b.shape()[1]];
    let mut scratch = rt::asarray((scratch_vec, shape.f(), device)).into_dim::<Ix2>();
    scratch.matmul_from(a.view().into_dim::<Ix2>().view(), b.view(), T::one(), T::zero());
}

#[test]
fn test_generic_matmul_under_lapack_driver() {
    let device = DeviceBLAS::default();
    let a = rt::asarray((vec![1.0, 2.0, 3.0, 4.0], [2, 2].f(), &device)).into_dim::<Ix2>();
    let b = rt::asarray((vec![1.0, 0.0, 0.0, 1.0], [2, 2].f(), &device)).into_dim::<Ix2>();

    // allocating entry (a = [[1, 3], [2, 4]] since the fill order is column-major)
    let c = generic_matmul(&a, &b);
    let c = c.to_contig(RowMajor).into_owned();
    assert_eq!(c.view().into_shape((-1,)).to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

    // scratch (matmul_from) entry
    let mut scratch = rt::zeros(([2, 2].f(), &device)).into_dim::<Ix2>().into_owned();
    generic_matmul_scratch(scratch.view_mut(), a.view(), b.view());
    let scratch = scratch.to_contig(RowMajor).into_owned();
    assert_eq!(scratch.view().into_shape((-1,)).to_vec(), vec![1.0, 3.0, 2.0, 4.0]);

    // dynamic-dimension scratch entry (batched style)
    let a_dyn = a.view().into_dim::<IxD>().into_owned();
    let mut scratch_vec = vec![0.0; 4];
    generic_matmul_dyn(&mut scratch_vec, &a_dyn, &b, &device);
    assert_eq!(scratch_vec, vec![1.0, 2.0, 3.0, 4.0]);
}

fn conj_elem<T: BlasFloat>(x: &mut MaybeUninit<T>) {
    unsafe { *x.assume_init_mut() = x.assume_init().conj() };
}

/// One representative call per member absorbed from the linalg tier, for a
/// real and a complex `T` (for the complex `T` the complex counterpart is `T`
/// itself). Returns the eig-style spectrum copy, the real vector copy, the
/// permutation, the vector norm and the rank.
#[allow(clippy::type_complexity)]
fn generic_linalg_members<B, T>(device: &B) -> Result<(Vec<Complex<T::Real>>, Vec<T::Real>, Vec<usize>, T::Real, usize)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let one = T::Real::one();
    let iota: Vec<T::Real> = (0..6).map(|i| T::Real::from_usize(i).unwrap()).collect();

    // complex counterpart of a real `T`: creation, 1-D assign from a strided
    // view, 1-D/2-D contiguity conversions, `Vec` raw
    let base: Tensor<Complex<T::Real>, B, Ix2> =
        rt::asarray((vec![Complex::new(one, one); 6], [3, 2].c(), device)).into_dim::<Ix2>().into_owned();
    let mut w: Tensor<Complex<T::Real>, B, Ix1> = rt::zeros(([3], device)).into_dim::<Ix1>();
    w.assign_f(base.i((.., 1)).into_dim::<Ix1>())?;
    let w = w.to_contig_f(ColMajor)?.into_owned();
    let vt: Tensor<Complex<T::Real>, B, Ix2> = base.t().to_contig_f(ColMajor)?.into_owned();
    assert_eq!(vt.shape().to_vec(), vec![2, 3]);

    // real vector through `into_contig_f` (eigh `w` slices); 1-D `T` likewise (gemv x/y, orgqr tau)
    // (`[3, 2]` row-major: column 1 is the strided `[1, 3, 5]`)
    let rbase: Tensor<T::Real, B, Ix2> = rt::asarray((iota.clone(), [3, 2].c(), device)).into_dim::<Ix2>().into_owned();
    let col: Tensor<T::Real, B, Ix1> = rbase.i((.., 1)).into_dim::<Ix1>().into_contig_f(ColMajor)?;
    let tbase: Tensor<T, B, Ix2> = rt::asarray((vec![T::one(); 4], [2, 2].c(), device)).into_dim::<Ix2>().into_owned();
    let tau: Tensor<T, B, Ix1> = tbase.i((.., 0)).into_dim::<Ix1>().into_contig_f(ColMajor)?;
    assert_eq!(tau.shape().to_vec(), vec![2]);

    // index dtypes: a `usize` permutation handed out through a `TensorMut` (ldl, lu);
    // the `bool` mask is produced by the comparison below (matrix_rank)
    let mut perm: Tensor<usize, B, Ix1> = rt::zeros(([3], device)).into_dim::<Ix1>();
    let mut perm_v: TensorMut<'_, usize, B, Ix1> = perm.view_mut();
    perm_v[[2]] = 7;

    // n-dim norm ops (vector_norm p-order path, matrix_rank tail)
    let a: Tensor<T, B, IxD> = rt::asarray((vec![T::one(); 4], [2, 2].c(), device));
    let nrm = rt::l2_norm_all_f(&a)?;
    let t = rt::pow_f(&rt::abs_f(&a)?, one + one)?;
    let t = rt::sign_f(rt::sum_with_args_f(&t, (-1, false))?)?;
    let smax = rt::max_with_args_f(&t, (-1, false))?;
    let _smin = rt::min_with_args_f(&t, (-1, false))?;
    let tol = &smax * (one + one);
    let rank = rt::greater_f(&t, &tol)?.sum_all_f()?;

    // in-place elementwise mutation of a matrix buffer (lu_solve conjugation)
    let mut y: Tensor<T, B, Ix2> = rt::asarray((vec![T::one(); 4], [2, 2].c(), device)).into_dim::<Ix2>();
    let mut conj = conj_elem::<T> as fn(&mut MaybeUninit<T>);
    op_muta_func(y.view_mut(), &mut conj)?;

    Ok((w.to_vec(), col.to_vec(), perm.to_vec(), nrm, rank))
}

#[test]
fn test_linalg_members_under_lapack_driver() {
    let device = DeviceBLAS::default();

    let (w, col, perm, nrm, rank) = generic_linalg_members::<_, f64>(&device).unwrap();
    assert_eq!(w, vec![Complex::new(1.0, 1.0); 3]);
    assert_eq!(col, vec![1.0, 3.0, 5.0]);
    assert_eq!(perm, vec![0, 0, 7]);
    assert!((nrm - 2.0).abs() < 1e-12);
    assert_eq!(rank, 0);

    let (w, col, perm, nrm, rank) = generic_linalg_members::<_, Complex<f64>>(&device).unwrap();
    assert_eq!(w, vec![Complex::new(1.0, 1.0); 3]);
    assert_eq!(col, vec![1.0, 3.0, 5.0]);
    assert_eq!(perm, vec![0, 0, 7]);
    assert!((nrm - 2.0).abs() < 1e-12);
    assert_eq!(rank, 0);
}
