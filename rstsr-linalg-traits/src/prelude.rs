pub mod rstsr_traits {
    pub use crate::traits::{
        cholesky::CholeskyAPI, eigh::EighAPI, inv::InvAPI, solve_general::SolveGeneralAPI,
        solve_symmetric::SolveSymmetricAPI, solve_triangular::SolveTriangularAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits::{
        cholesky::{cholesky, cholesky_f},
        eigh::{eigh, eigh_f},
        inv::{inv, inv_f},
        solve_general::{solve_general, solve_general_f},
        solve_symmetric::{solve_symmetric, solve_symmetric_f},
        solve_triangular::{solve_triangular, solve_triangular_f},
    };
}

pub mod rstsr_structs {
    pub use crate::traits::eigh::{EighArgs, EighArgs_, EighArgs_Builder};
}
