use rstsr_core::prelude::rt::Result;

pub trait SolveSymmetricAPI<Inp> {
    type Out;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out>;
    fn solve_symmetric(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_symmetric_f(args).unwrap()
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
