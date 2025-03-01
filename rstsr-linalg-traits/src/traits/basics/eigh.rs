use derive_builder::Builder;
use rstsr_blas_traits::prelude::BlasFloat;
use rstsr_core::prelude_dev::*;

pub trait LinalgEighAPI<Inp>: Sized {
    type Out;
    fn eigh_f(args: Self) -> Result<Self::Out>;
    fn eigh(args: Self) -> Self::Out {
        Self::eigh_f(args).unwrap()
    }
}

pub fn eigh_f<Args, Inp>(args: Args) -> Result<<Args as LinalgEighAPI<Inp>>::Out>
where
    Args: LinalgEighAPI<Inp>,
{
    Args::eigh_f(args)
}

pub fn eigh<Args, Inp>(args: Args) -> <Args as LinalgEighAPI<Inp>>::Out
where
    Args: LinalgEighAPI<Inp>,
{
    Args::eigh(args)
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
