use crate::traits_def::{EighAPI, EighResult};
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;
use rstsr_dtype_traits::ReImAPI;

pub fn faer_impl_standard_eigh_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<(Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    // TODO: It seems faer is suspeciously slow on eigh function?
    // However, tests shows that results are correct.

    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }

    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = a.into_faer();
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    // eigen value computation
    let result = faer_a
        .self_adjoint_eigen(faer_uplo)
        .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;

    // convert eigenvalues to real
    let eigenvalues: TensorView<T, DeviceFaer, _> = result.S().column_vector().into_rstsr();
    let eigenvalues = eigenvalues.mapv(|v| T::real_part_impl(&v));
    let eigenvectors = result.U().into_rstsr().into_contig(device.default_order());

    // restore parallel mode
    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    Ok((eigenvalues, eigenvectors))
}

pub fn faer_impl_generalized_eigh_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    b: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
    itype: i32,
) -> Result<(Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField,
{
    // check sanity
    rstsr_assert!(a.device().same_device(b.device()), DeviceMismatch)?;
    rstsr_pattern!(itype, 1..=3, InvalidValue, "itype in generalized eigen must be 1, 2 or 3")?;
    rstsr_assert_eq!(a.nrow(), a.ncol(), InvalidLayout, "Matrix a must be square.")?;

    // set parallel mode
    let device = a.device().clone();

    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = if let Some(pool) = pool {
        let faer_par = Par::rayon(pool.current_num_threads());
        faer::set_global_parallelism(faer_par);
        faer_par
    } else {
        Par::Seq
    };

    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    // perform symmetrize on matrix a
    // TODO: a better implementation
    let mut a = a.to_owned();
    let n = a.nrow();
    match uplo {
        Lower => {
            for i in 0..n {
                for j in 0..i {
                    a[[j, i]] = T::conj_impl(&a[[i, j]]);
                }
                a[[i, i]] = T::from_real_impl(&T::real_part_impl(&a[[i, i]]));
            }
        },
        Upper => {
            for i in 0..n {
                for j in (i + 1)..n {
                    a[[j, i]] = T::conj_impl(&a[[i, j]]);
                }
                a[[i, i]] = T::from_real_impl(&T::real_part_impl(&a[[i, i]]));
            }
        },
    }

    // perform cholesky on matrix b
    let b = b.into_faer();
    let b_llt = b.llt(faer_uplo).map_err(|e| rstsr_error!(FaerError, "Faer Cholesky error: {e:?}"))?;
    let l = b_llt.L();

    let result = match itype {
        1 => {
            // inv(l) @ a @ inv(l.t.conj)
            let mut a = a.view_mut().into_faer();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.as_ref(), a.as_mut(), faer_par);
            let mut a = a.transpose_mut();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.conjugate(), a.as_mut(), faer_par);
            let a = a.transpose_mut();

            // eig(a)
            let eig_a = a
                .self_adjoint_eigen(faer_uplo)
                .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
            let e = eig_a.S().column_vector().into_rstsr().mapv(|v| T::real_part_impl(&v));

            // inv(l.t.conj) @ c
            let mut c = eig_a.U().to_owned();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l.adjoint(), c.as_mut(), faer_par);
            let c = c.into_rstsr().into_contig(device.default_order());

            (e, c)
        },
        2 | 3 => {
            // inv(l)
            let mut l_inv = Mat::identity(n, n);
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(l.as_ref(), l_inv.as_mut(), faer_par);

            // l.t.conj @ a @ l
            let mut a = a.view_mut().into_faer();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_inv.adjoint(), a.as_mut(), faer_par);
            let mut a = a.transpose_mut();
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(l_inv.transpose(), a.as_mut(), faer_par);
            let a = a.transpose_mut();

            // eig(a)
            let eig_a = a
                .self_adjoint_eigen(faer_uplo)
                .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
            let e = eig_a.S().column_vector().into_rstsr().mapv(|v| T::real_part_impl(&v));

            let mut c = eig_a.U().to_owned();
            match itype {
                2 => faer::linalg::triangular_solve::solve_upper_triangular_in_place(l.adjoint(), c.as_mut(), faer_par),
                3 => faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    l_inv.as_ref(),
                    c.as_mut(),
                    faer_par,
                ),
                _ => unreachable!(),
            };
            let c = c.into_rstsr().into_contig(device.default_order());

            (e, c)
        },
        _ => unreachable!(),
    };

    // restore parallel mode
    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    Ok(result)
}

/* #region standard eigh */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = faer_impl_standard_eigh_f(a.view(), uplo)?;
        let result = EighResult {
            eigenvalues: result.0.into_dim::<IxD>().into_dim::<D::SmallerOne>(),
            eigenvectors: result.1.into_owned().into_dim::<IxD>().into_dim::<D>(),
        };
        Ok(result)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EighAPI::<DeviceFaer>::eigh_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for Tr
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        EighAPI::<DeviceFaer>::eigh_f((a, None))
    }
}

/* #endregion */

/* #region generalized eig */

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB, FlagUpLo, i32)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo, eig_type) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(eig_type, 1..=3, InvalidLayout, "Only eig_type = 1, 2, or 3 allowed.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let (vals, vecs) = faer_impl_generalized_eigh_f(a_view.view(), b_view.view(), Some(uplo), eig_type)?;
        let vals = vals.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        let vecs = vecs.into_owned().into_dim::<IxD>().into_dim::<D>();
        Ok(EighResult { eigenvalues: vals, eigenvectors: vecs })
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB, FlagUpLo)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        EighAPI::<DeviceFaer>::eigh_f((a, b, uplo, 1))
    }
}

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> EighAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        EighAPI::<DeviceFaer>::eigh_f((a, b, uplo, 1))
    }
}

/* #endregion */
