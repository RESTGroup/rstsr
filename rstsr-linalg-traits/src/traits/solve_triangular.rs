use rstsr_core::prelude::rt::Result;

pub trait SolveTriangularAPI<Inp> {
    type Out;
    fn solve_triangular_f(args: Self) -> Result<Self::Out>;
    fn solve_triangular(args: Self) -> Self::Out
    where
        Self: Sized,
    {
        Self::solve_triangular_f(args).unwrap()
    }
}

pub fn solve_triangular_f<Args, Inp>(args: Args) -> Result<<Args as SolveTriangularAPI<Inp>>::Out>
where
    Args: SolveTriangularAPI<Inp>,
{
    Args::solve_triangular_f(args)
}

pub fn solve_triangular<Args, Inp>(args: Args) -> <Args as SolveTriangularAPI<Inp>>::Out
where
    Args: SolveTriangularAPI<Inp>,
{
    Args::solve_triangular(args)
}
