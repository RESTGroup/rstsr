/* #region flags of cblas */

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasLayout {
    RowMajor = 101,
    ColMajor = 102,
}

pub type CblasOrder = CblasLayout;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasUplo {
    Upper = 121,
    Lower = 122,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasDiag {
    NonUnit = 131,
    Unit = 132,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasSide {
    Left = 141,
    Right = 142,
}

pub type CBLAS_ORDER = u32;
pub type CBLAS_TRANSPOSE = u32;
pub type CBLAS_UPLO = u32;
pub type CBLAS_DIAG = u32;
pub type CBLAS_SIDE = u32;

/* #endregion */

/* #region flag into */

use rstsr_core::flags::{FlagDiag, FlagOrder, FlagSide, FlagTrans, FlagUpLo};

impl From<FlagTrans> for CblasTranspose {
    fn from(flag: FlagTrans) -> Self {
        match flag {
            FlagTrans::N => CblasTranspose::NoTrans,
            FlagTrans::T => CblasTranspose::Trans,
            FlagTrans::C => CblasTranspose::ConjTrans,
            _ => panic!("Invalid flag for trans"),
        }
    }
}

impl From<FlagUpLo> for CblasUplo {
    fn from(flag: FlagUpLo) -> Self {
        match flag {
            FlagUpLo::U => CblasUplo::Upper,
            FlagUpLo::L => CblasUplo::Lower,
            _ => panic!("Invalid flag for uplo"),
        }
    }
}

impl From<FlagDiag> for CblasDiag {
    fn from(flag: FlagDiag) -> Self {
        match flag {
            FlagDiag::U => CblasDiag::Unit,
            FlagDiag::N => CblasDiag::NonUnit,
            _ => panic!("Invalid flag for diag"),
        }
    }
}

impl From<FlagSide> for CblasSide {
    fn from(flag: FlagSide) -> Self {
        match flag {
            FlagSide::L => CblasSide::Left,
            FlagSide::R => CblasSide::Right,
            _ => panic!("Invalid flag for side"),
        }
    }
}

impl From<FlagOrder> for CblasLayout {
    fn from(order: FlagOrder) -> Self {
        match order {
            RowMajor => CblasOrder::RowMajor,
            ColMajor => CblasOrder::ColMajor,
        }
    }
}

/* #endregion */
