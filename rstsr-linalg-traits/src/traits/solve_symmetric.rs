use rstsr_core::error::Result;

pub trait LinalgSolveSymmetricAPI<Inp>: Sized {
    type Out;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out>;
    fn solve_symmetric(args: Self) -> Self::Out {
        Self::solve_symmetric_f(args).unwrap()
    }
}

pub fn solve_symmetric_f<Args, Inp>(args: Args) -> Result<<Args as LinalgSolveSymmetricAPI<Inp>>::Out>
where
    Args: LinalgSolveSymmetricAPI<Inp>,
{
    Args::solve_symmetric_f(args)
}

pub fn solve_symmetric<Args, Inp>(args: Args) -> <Args as LinalgSolveSymmetricAPI<Inp>>::Out
where
    Args: LinalgSolveSymmetricAPI<Inp>,
{
    Args::solve_symmetric(args)
}
