use derive_builder::Builder;
use rstsr_blas_traits::prelude::BlasFloat;
use rstsr_common::error::{Error, Result};
use rstsr_core::prelude::*;

/* #region cholesky */

pub trait CholeskyAPI<Inp> {
    type Out;
    fn cholesky_f(self) -> Result<Self::Out>;
    fn cholesky(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::cholesky_f(self).unwrap()
    }
}

pub fn cholesky_f<Args, Inp>(args: Args) -> Result<<Args as CholeskyAPI<Inp>>::Out>
where
    Args: CholeskyAPI<Inp>,
{
    Args::cholesky_f(args)
}

pub fn cholesky<Args, Inp>(args: Args) -> <Args as CholeskyAPI<Inp>>::Out
where
    Args: CholeskyAPI<Inp>,
{
    Args::cholesky(args)
}

/* #endregion */

/* #region eigh */

pub trait EighAPI<Inp> {
    type Out;
    fn eigh_f(self) -> Result<Self::Out>;
    fn eigh(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::eigh_f(self).unwrap()
    }
}

pub fn eigh_f<Args, Inp>(args: Args) -> Result<<Args as EighAPI<Inp>>::Out>
where
    Args: EighAPI<Inp>,
{
    Args::eigh_f(args)
}

pub fn eigh<Args, Inp>(args: Args) -> <Args as EighAPI<Inp>>::Out
where
    Args: EighAPI<Inp>,
{
    Args::eigh(args)
}

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

/* #region inv */

pub trait InvAPI<Inp> {
    type Out;
    fn inv_f(self) -> Result<Self::Out>;
    fn inv(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::inv_f(self).unwrap()
    }
}

pub fn inv_f<Args, Inp>(args: Args) -> Result<<Args as InvAPI<Inp>>::Out>
where
    Args: InvAPI<Inp>,
{
    Args::inv_f(args)
}

pub fn inv<Args, Inp>(args: Args) -> <Args as InvAPI<Inp>>::Out
where
    Args: InvAPI<Inp>,
{
    Args::inv(args)
}

/* #endregion */

/* #region solve_general */

pub trait SolveGeneralAPI<Inp> {
    type Out;
    fn solve_general_f(self) -> Result<Self::Out>;
    fn solve_general(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_general_f(self).unwrap()
    }
}

pub fn solve_general_f<Args, Inp>(args: Args) -> Result<<Args as SolveGeneralAPI<Inp>>::Out>
where
    Args: SolveGeneralAPI<Inp>,
{
    Args::solve_general_f(args)
}

pub fn solve_general<Args, Inp>(args: Args) -> <Args as SolveGeneralAPI<Inp>>::Out
where
    Args: SolveGeneralAPI<Inp>,
{
    Args::solve_general(args)
}

/* #endregion */

/* #region solve_symmetric */

pub trait SolveSymmetricAPI<Inp> {
    type Out;
    fn solve_symmetric_f(self) -> Result<Self::Out>;
    fn solve_symmetric(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_symmetric_f(self).unwrap()
    }
}

pub fn solve_symmetric_f<Args, Inp>(args: Args) -> Result<<Args as SolveSymmetricAPI<Inp>>::Out>
where
    Args: SolveSymmetricAPI<Inp>,
{
    Args::solve_symmetric_f(args)
}

pub fn solve_symmetric<Args, Inp>(args: Args) -> <Args as SolveSymmetricAPI<Inp>>::Out
where
    Args: SolveSymmetricAPI<Inp>,
{
    Args::solve_symmetric(args)
}

/* #endregion */

/* #region solve_triangular */

pub trait SolveTriangularAPI<Inp> {
    type Out;
    fn solve_triangular_f(self) -> Result<Self::Out>;
    fn solve_triangular(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_triangular_f(self).unwrap()
    }
}

pub fn solve_triangular_f<Args, Inp>(args: Args) -> Result<<Args as SolveTriangularAPI<Inp>>::Out>
where
    Args: SolveTriangularAPI<Inp>,
{
    Args::solve_triangular_f(args)
}

pub fn solve_triangular<Args, Inp>(args: Args) -> <Args as SolveTriangularAPI<Inp>>::Out
where
    Args: SolveTriangularAPI<Inp>,
{
    Args::solve_triangular(args)
}

/* #endregion */
