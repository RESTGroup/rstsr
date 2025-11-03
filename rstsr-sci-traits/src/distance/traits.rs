use rstsr_core::prelude_dev::*;

pub trait CDistAPI<Inp> {
    type Out;

    fn cdist_f(self) -> Result<Self::Out>;
    fn cdist(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::cdist_f(self).rstsr_unwrap()
    }
}

pub fn cdist<Args, Inp>(args: Args) -> Args::Out
where
    Args: CDistAPI<Inp>,
{
    args.cdist()
}

pub fn cdist_f<Args, Inp>(args: Args) -> Result<Args::Out>
where
    Args: CDistAPI<Inp>,
{
    args.cdist_f()
}
