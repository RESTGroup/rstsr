pub mod rstsr_traits {
    pub use crate::traits_def::{
        ChoFactorAPI, ChoSolveAPI, CholeskyAPI, DetAPI, EigAPI, EighAPI, EigvalsAPI, EigvalshAPI, InvAPI, LdlAPI,
        LstsqAPI, LuAPI, LuFactorAPI, LuSolveAPI, MatrixNormAPI, MatrixPowerAPI, MatrixRankAPI, PinvAPI, QrAPI,
        SLogDetAPI, SVDvalsAPI, SolveGeneralAPI, SolveSymmetricAPI, SolveTriangularAPI, VectorNormAPI, SVDAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        cho_factor, cho_factor_f, cho_solve, cho_solve_f, cholesky, cholesky_f, det, det_f, eig, eig_f, eigh, eigh_f,
        eigvals, eigvals_f, eigvalsh, eigvalsh_f, inv, inv_f, ldl, ldl_f, lstsq, lstsq_f, lu, lu_f, lu_factor,
        lu_factor_f, lu_solve, lu_solve_f, matrix_norm, matrix_norm_f, matrix_power, matrix_power_f, matrix_rank,
        matrix_rank_f, pinv, pinv_f, qr, qr_f, slogdet, slogdet_f, solve_general, solve_general_f, solve_symmetric,
        solve_symmetric_f, solve_triangular, solve_triangular_f, svd, svd_f, svdvals, svdvals_f, vector_norm,
        vector_norm_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{
        ChoFactorResult, EigResult, EigSide, EighArgs, EighArgs_, EighArgs_Builder, EighResult, Homogeneous, LdlResult,
        LstsqResult, LuFactorResult, LuResult, MatrixNormArgs, MatrixRankRtol, NormOrd, PinvRtol, QrMode, QrResult,
        SLogDetResult, SVDArgs, SVDArgs_, SVDArgs_Builder, SVDResult, VectorNormArgs,
    };
}
