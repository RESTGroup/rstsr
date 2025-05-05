use rstsr_core::prelude::rt::Result;

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
