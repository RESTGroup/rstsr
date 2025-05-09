pub mod rstsr_traits {
    pub use crate::traits_def::{
        CholeskyAPI, DetAPI, EighAPI, InvAPI, SLogDetAPI, SolveGeneralAPI, SolveSymmetricAPI,
        SolveTriangularAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        cholesky, cholesky_f, det, det_f, eigh, eigh_f, inv, inv_f, slogdet, slogdet_f,
        solve_general, solve_general_f, solve_symmetric, solve_symmetric_f, solve_triangular,
        solve_triangular_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{EighArgs, EighArgs_, EighArgs_Builder, EighResult, SLogDetResult};
}
