use rstsr_core::error::Result;

pub trait LinalgSolveTriangularAPI<Inp> {
    type Out;
    fn solve_triangular_f(args: Self) -> Result<Self::Out>;
    fn solve_triangular(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_triangular_f(args).unwrap()
    }
}

pub fn solve_triangular_f<Args, Inp>(
    args: Args,
) -> Result<<Args as LinalgSolveTriangularAPI<Inp>>::Out>
where
    Args: LinalgSolveTriangularAPI<Inp>,
{
    Args::solve_triangular_f(args)
}

pub fn solve_triangular<Args, Inp>(args: Args) -> <Args as LinalgSolveTriangularAPI<Inp>>::Out
where
    Args: LinalgSolveTriangularAPI<Inp>,
{
    Args::solve_triangular(args)
}
