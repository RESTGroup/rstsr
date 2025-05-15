use derive_builder::Builder;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::BlasFloat;
use rstsr_common::error::{Error, Result};
use rstsr_core::prelude::*;

/* #region trait and fn definitions */

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
)]
pub trait LinalgAPI<Inp> {
    type Out;
    fn func_f(self) -> Result<Self::Out>;
    fn func(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::func_f(self).unwrap()
    }
}

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
)]
pub fn func_f<Args, Inp>(args: Args) -> Result<<Args as LinalgAPI<Inp>>::Out>
where
    Args: LinalgAPI<Inp>,
{
    Args::func_f(args)
}

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
)]
pub fn func<Args, Inp>(args: Args) -> <Args as LinalgAPI<Inp>>::Out
where
    Args: LinalgAPI<Inp>,
{
    Args::func(args)
}

/* #endregion */

/* #region eigh */

pub struct EighResult<W, V> {
    pub eigenvalues: W,
    pub eigenvectors: V,
}

impl<W, V> From<(W, V)> for EighResult<W, V> {
    fn from((vals, vecs): (W, V)) -> Self {
        Self { eigenvalues: vals, eigenvectors: vecs }
    }
}

impl<W, V> From<EighResult<W, V>> for (W, V) {
    fn from(eigh_result: EighResult<W, V>) -> Self {
        (eigh_result.eigenvalues, eigh_result.eigenvectors)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct EighArgs_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into, strip_option), default = "None")]
    pub b: Option<TensorReference<'b, T, B, Ix2>>,

    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
    #[builder(setter(into), default = false)]
    pub eigvals_only: bool,
    #[builder(setter(into), default = 1)]
    pub eig_type: i32,
    #[builder(setter(into, strip_option), default = "None")]
    pub subset_by_index: Option<(usize, usize)>,
    #[builder(setter(into, strip_option), default = "None")]
    pub subset_by_value: Option<(T::Real, T::Real)>,
    #[builder(setter(into, strip_option), default = "None")]
    pub driver: Option<&'static str>,
}

pub type EighArgs<'a, 'b, B, T> = EighArgs_Builder<'a, 'b, B, T>;

/* #endregion */

/* #region slogdet */

pub struct SLogDetResult<T>
where
    T: BlasFloat,
{
    pub sign: T,
    pub logabsdet: T::Real,
}

impl<T> From<(T, T::Real)> for SLogDetResult<T>
where
    T: BlasFloat,
{
    fn from((sign, logabsdet): (T, T::Real)) -> Self {
        Self { sign, logabsdet }
    }
}

impl<T> From<SLogDetResult<T>> for (T, T::Real)
where
    T: BlasFloat,
{
    fn from(slogdet_result: SLogDetResult<T>) -> Self {
        (slogdet_result.sign, slogdet_result.logabsdet)
    }
}

/* #endregion */

/* #region svd */

pub struct SVDResult<U, S, Vt> {
    pub u: U,
    pub s: S,
    pub vt: Vt,
}

impl<U, S, Vt> From<(U, S, Vt)> for SVDResult<U, S, Vt> {
    fn from((u, s, vt): (U, S, Vt)) -> Self {
        Self { u, s, vt }
    }
}

impl<U, S, Vt> From<SVDResult<U, S, Vt>> for (U, S, Vt) {
    fn from(svd_result: SVDResult<U, S, Vt>) -> Self {
        (svd_result.u, svd_result.s, svd_result.vt)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SVDArgs_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into), default = "Some(true)")]
    pub full_matrices: Option<bool>,
    #[builder(setter(into), default = "None")]
    pub driver: Option<&'static str>,
}

pub type SVDArgs<'a, B, T> = SVDArgs_Builder<'a, B, T>;

/* #endregion */
