pub mod rstsr_traits {
    pub use crate::traits_def::{
        CholeskyAPI, DetAPI, EigAPI, EighAPI, EigvalsAPI, EigvalshAPI, InvAPI, LstsqAPI, MatrixNormAPI, MatrixPowerAPI,
        MatrixRankAPI, PinvAPI, QrAPI, SLogDetAPI, SVDvalsAPI, SolveGeneralAPI, SolveSymmetricAPI, SolveTriangularAPI,
        VectorNormAPI, SVDAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        cholesky, cholesky_f, det, det_f, eig, eig_f, eigh, eigh_f, eigvals, eigvals_f, eigvalsh, eigvalsh_f, inv,
        inv_f, lstsq, lstsq_f, matrix_norm, matrix_norm_f, matrix_power, matrix_power_f, matrix_rank, matrix_rank_f,
        pinv, pinv_f, qr, qr_f, slogdet, slogdet_f, solve_general, solve_general_f, solve_symmetric, solve_symmetric_f,
        solve_triangular, solve_triangular_f, svd, svd_f, svdvals, svdvals_f, vector_norm, vector_norm_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{
        EigResult, EigSide, EighArgs, EighArgs_, EighArgs_Builder, EighResult, Homogeneous, LstsqResult,
        MatrixNormArgs, MatrixRankRtol, NormOrd, PinvRtol, QrMode, QrResult, SLogDetResult, SVDArgs, SVDArgs_,
        SVDArgs_Builder, SVDResult, VectorNormArgs,
    };
}
