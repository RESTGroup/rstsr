pub mod rstsr_traits {
    pub use crate::traits_def::{
        CholeskyAPI, DetAPI, EigAPI, EighAPI, EigvalsAPI, EigvalshAPI, InvAPI, PinvAPI, SLogDetAPI, SVDvalsAPI,
        SolveGeneralAPI, SolveSymmetricAPI, SolveTriangularAPI, QRAPI, SVDAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        cholesky, cholesky_f, det, det_f, eig, eig_f, eigh, eigh_f, eigvals, eigvals_f, eigvalsh, eigvalsh_f, inv,
        inv_f, pinv, pinv_f, qr, qr_f, slogdet, slogdet_f, solve_general, solve_general_f, solve_symmetric,
        solve_symmetric_f, solve_triangular, solve_triangular_f, svd, svd_f, svdvals, svdvals_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{
        EigArgs, EigArgs_, EigArgs_Builder, EigResult, EighArgs, EighArgs_, EighArgs_Builder, EighResult, PinvResult,
        QRResult, SLogDetResult, SVDArgs, SVDArgs_, SVDArgs_Builder, SVDResult,
    };
}
