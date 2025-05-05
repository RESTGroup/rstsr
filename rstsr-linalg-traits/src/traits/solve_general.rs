use rstsr_core::prelude::rt::Result;

pub trait SolveGeneralAPI<Inp> {
    type Out;
    fn solve_general_f(args: Self) -> Result<Self::Out>;
    fn solve_general(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_general_f(args).unwrap()
    }
}

pub fn solve_general_f<Args, Inp>(args: Args) -> Result<<Args as SolveGeneralAPI<Inp>>::Out>
where
    Args: SolveGeneralAPI<Inp>,
{
    Args::solve_general_f(args)
}

pub fn solve_general<Args, Inp>(args: Args) -> <Args as SolveGeneralAPI<Inp>>::Out
where
    Args: SolveGeneralAPI<Inp>,
{
    Args::solve_general(args)
}
