use derive_builder::Builder;
use rstsr_blas_traits::prelude::BlasFloat;
use rstsr_core::prelude_dev::*;

pub trait EighAPI<Inp> {
    type Out;
    fn eigh_f(args: Self) -> Result<Self::Out>;
    fn eigh(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::eigh_f(args).unwrap()
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

    #[builder(setter(into), default = "Lower")]
    pub uplo: FlagUpLo,
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
