use rstsr_core::error::Result;

pub trait LinalgCholeskyAPI<Inp> {
    type Out;
    fn cholesky_f(args: Self) -> Result<Self::Out>;
    fn cholesky(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::cholesky_f(args).unwrap()
    }
}

pub fn cholesky_f<Args, Inp>(args: Args) -> Result<<Args as LinalgCholeskyAPI<Inp>>::Out>
where
    Args: LinalgCholeskyAPI<Inp>,
{
    Args::cholesky_f(args)
}

pub fn cholesky<Args, Inp>(args: Args) -> <Args as LinalgCholeskyAPI<Inp>>::Out
where
    Args: LinalgCholeskyAPI<Inp>,
{
    Args::cholesky(args)
}
