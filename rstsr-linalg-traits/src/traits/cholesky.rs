use rstsr_core::prelude::rt::Result;

pub trait CholeskyAPI<Inp> {
    type Out;
    fn cholesky_f(args: Self) -> Result<Self::Out>;
    fn cholesky(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::cholesky_f(args).unwrap()
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
