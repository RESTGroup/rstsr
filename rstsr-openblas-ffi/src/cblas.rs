use crate::ffi;

/* #region flags of cblas */

#[repr(u32)]
pub enum CblasLayout {
    RowMajor = ffi::cblas::CBLAS_ORDER_CblasRowMajor as u32,
    ColMajor = ffi::cblas::CBLAS_ORDER_CblasColMajor as u32,
}

#[repr(u32)]
pub enum CblasTranspose {
    NoTrans = ffi::cblas::CBLAS_TRANSPOSE_CblasNoTrans as u32,
    Trans = ffi::cblas::CBLAS_TRANSPOSE_CblasTrans as u32,
    ConjTrans = ffi::cblas::CBLAS_TRANSPOSE_CblasConjTrans as u32,
    ConjNoTrans = ffi::cblas::CBLAS_TRANSPOSE_CblasConjNoTrans as u32,
}

#[repr(u32)]
pub enum CblasUplo {
    Upper = ffi::cblas::CBLAS_UPLO_CblasUpper as u32,
    Lower = ffi::cblas::CBLAS_UPLO_CblasLower as u32,
}

#[repr(u32)]
pub enum CblasDiag {
    NonUnit = ffi::cblas::CBLAS_DIAG_CblasNonUnit as u32,
    Unit = ffi::cblas::CBLAS_DIAG_CblasUnit as u32,
}

#[repr(u32)]
pub enum CblasSide {
    Left = ffi::cblas::CBLAS_SIDE_CblasLeft as u32,
    Right = ffi::cblas::CBLAS_SIDE_CblasRight as u32,
}

pub use CblasDiag::{NonUnit, Unit};
pub use CblasLayout::{ColMajor, RowMajor};
pub use CblasSide::{Left, Right};
pub use CblasTranspose::{ConjNoTrans, ConjTrans, NoTrans, Trans};
pub use CblasUplo::{Lower, Upper};

/* #endregion */
