//! General (non-Hermitian) eigendecomposition on the faer device (the
//! array-API `linalg.eig`).
//!
//! Unlike [`eigh`](crate::faer_impl::eigh) this takes the full matrix (no
//! `uplo`) and returns **complex** results for every input: a real matrix can
//! have a complex spectrum, so the output dtype is the complex counterpart of
//! the input's real type (`Complex<T::Real>`). Eigenvalues are returned in
//! faer's (LAPACK-like) order — no sorting — and each eigenvector column is
//! rescaled to unit 2-norm (faer does not normalize; numpy and the array API
//! expect unit-norm columns). The array API grades neither order nor values.
//!
//! The result is the shared [`EigResult`] with the **right** eigenvectors
//! only: `left` is always `None` (faer computes no left eigenvectors) and
//! `right` is `Some(vr)`, so `.into()` gives the `(w, vr)` pair. The generalized
//! problem, the left/both sides and the homogeneous spectrum are BLAS-device
//! forms (see [`eig`](crate::traits_def::eig)).

use crate::faer_impl::batch::{batch_and_matrix_shape, map_stack_slices2, stack_shape, with_parallel};
use crate::traits_def::{EigAPI, EigResult};
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use num::Complex;
use rstsr_core::prelude_dev::*;

/// Rescale each eigenvector column to unit 2-norm — faer's `eigen()` does not
/// normalize (its back-substitution fixes the pivot at `1`), while numpy and
/// the array API expect unit-norm columns.
fn normalize_columns<T>(mut v: Tensor<Complex<T::Real>, DeviceFaer, Ix2>) -> Tensor<Complex<T::Real>, DeviceFaer, Ix2>
where
    T: ComplexField,
{
    let n = v.shape()[1];
    for j in 0..n {
        let mut norm_sq = Complex::<T::Real>::abs2_impl(&v[[0, j]]);
        for i in 1..n {
            norm_sq += Complex::<T::Real>::abs2_impl(&v[[i, j]]);
        }
        let norm = Complex::<T::Real>::sqrt_impl(&Complex::<T::Real>::from_real_impl(&norm_sq));
        for i in 0..n {
            v[[i, j]] = v[[i, j]].clone() / norm.clone();
        }
    }
    v
}

/// Eigendecomposition of a single 2-D matrix (no parallel-mode handling).
fn faer_eig_ix2<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
) -> Result<(Tensor<Complex<T::Real>, DeviceFaer, Ix1>, Tensor<Complex<T::Real>, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let result = a.into_faer().eigen().map_err(|e| rstsr_error!(FaerError, "Faer Eigen error: {e:?}"))?;

    // `into_rstsr` homes the results on `DeviceFaer::default()`, so each device
    // must be changed back to the input's
    let eigenvectors = result.U().into_rstsr().into_contig(device.default_order()).change_device_f(&device)?;
    let eigenvalues = result.S().column_vector().into_rstsr().to_owned().change_device_f(&device)?;
    let eigenvectors = normalize_columns::<T>(eigenvectors);

    Ok((eigenvalues, eigenvectors))
}

pub fn faer_impl_eig_f<T>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
) -> Result<(Tensor<Complex<T::Real>, DeviceFaer, IxD>, Tensor<Complex<T::Real>, DeviceFaer, IxD>)>
where
    T: ComplexField,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;
    rstsr_assert_eq!(m, n, InvalidLayout, "eig: the matrix must be square, got {m}x{n}")?;

    // a zero-size matrix axis or batch yields empty outputs (faer is not called)
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        let w = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
        let v = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
        return Ok((w, v));
    }

    with_parallel(&device, || {
        let mut w = zeros_f((stack_shape(&batch_shape, &[m], order), &device))?;
        let mut v = zeros_f((stack_shape(&batch_shape, &[m, m], order), &device))?;
        map_stack_slices2::<T, Complex<T::Real>, Ix1, Complex<T::Real>, Ix2, _>(
            a,
            w.view_mut(),
            v.view_mut(),
            order,
            |a_slice, mut w_slice, mut v_slice| {
                let (ew, ev) = faer_eig_ix2(a_slice)?;
                w_slice.assign_f(ew)?;
                v_slice.assign_f(ev)
            },
        )?;
        Ok((w, v))
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceFaer, D::SmallerOne>,
        Option<Tensor<Complex<T::Real>, DeviceFaer, D>>,
        Option<Tensor<Complex<T::Real>, DeviceFaer, D>>,
    >;
    fn eig_f(self) -> Result<Self::Out> {
        let a = self;
        let (w, v) = faer_impl_eig_f(a.view().to_dyn())?;
        Ok(EigResult { eigenvalues: w.into_dim::<D::SmallerOne>(), left: None, right: Some(v.into_dim::<D>()) })
    }
}
