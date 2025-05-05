use rstsr_core::prelude::rt::Result;

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
