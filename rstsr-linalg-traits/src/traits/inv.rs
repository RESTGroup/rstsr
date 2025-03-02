use rstsr_core::error::Result;

pub trait LinalgInvAPI<Inp>: Sized {
    type Out;
    fn inv_f(args: Self) -> Result<Self::Out>;
    fn inv(args: Self) -> Self::Out {
        Self::inv_f(args).unwrap()
    }
}

pub fn inv_f<Args, Inp>(args: Args) -> Result<<Args as LinalgInvAPI<Inp>>::Out>
where
    Args: LinalgInvAPI<Inp>,
{
    Args::inv_f(args)
}

pub fn inv<Args, Inp>(args: Args) -> <Args as LinalgInvAPI<Inp>>::Out
where
    Args: LinalgInvAPI<Inp>,
{
    Args::inv(args)
}
