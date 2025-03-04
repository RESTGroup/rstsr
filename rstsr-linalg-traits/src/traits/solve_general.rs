use rstsr_core::error::Result;

pub trait LinalgSolveGeneralAPI<Inp> {
    type Out;
    fn solve_general_f(args: Self) -> Result<Self::Out>;
    fn solve_general(args: Self) -> Self::Out
    where
        Self: Sized, {
        Self::solve_general_f(args).unwrap()
    }
}

pub fn solve_general_f<Args, Inp>(args: Args) -> Result<<Args as LinalgSolveGeneralAPI<Inp>>::Out>
where
    Args: LinalgSolveGeneralAPI<Inp>,
{
    Args::solve_general_f(args)
}

pub fn solve_general<Args, Inp>(args: Args) -> <Args as LinalgSolveGeneralAPI<Inp>>::Out
where
    Args: LinalgSolveGeneralAPI<Inp>,
{
    Args::solve_general(args)
}
