//! Flags for the crate.

use crate::prelude_dev::*;
use core::ffi::c_char;
use rstsr_lapack_ffi::cblas::{CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO};

/* #region changeable default */

pub trait ChangeableDefault {
    /// # Safety
    ///
    /// This function changes static mutable variable.
    /// It is better applying cargo feature instead of using this function.
    unsafe fn change_default(val: Self);
    fn get_default() -> Self;
}

macro_rules! impl_changeable_default {
    ($struct:ty, $val:ident, $default:expr) => {
        static mut $val: $struct = $default;

        impl ChangeableDefault for $struct {
            unsafe fn change_default(val: Self) {
                $val = val;
            }

            fn get_default() -> Self {
                return unsafe { $val };
            }
        }

        impl Default for $struct
        where
            Self: ChangeableDefault,
        {
            fn default() -> Self {
                <$struct>::get_default()
            }
        }
    };
}

/* #endregion */

/* #region FlagOrder */

/// The order of the tensor.
///
/// # Default
///
/// Default order depends on cargo feature `f_prefer`.
/// If `f_prefer` is set, then [`FlagOrder::F`] is applied as default;
/// otherwise [`FlagOrder::C`] is applied as default.
///
/// # IMPORTANT NOTE
///
/// F-prefer is not a stable feature currently! We develop only in C-prefer
/// currently.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagOrder {
    /// row-major order.
    C = 101,
    /// column-major order.
    F = 102,
}

#[allow(non_upper_case_globals)]
impl FlagOrder {
    pub const RowMajor: Self = FlagOrder::C;
    pub const ColMajor: Self = FlagOrder::F;
}

#[allow(clippy::derivable_impls)]
impl Default for FlagOrder {
    fn default() -> Self {
        if cfg!(feature = "col_major") {
            return FlagOrder::F;
        } else {
            return FlagOrder::C;
        }
    }
}

/* #endregion */

/* #region TensorIterOrder */

/// The policy of the tensor iterator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIterOrder {
    /// Row-major order.
    ///
    /// - absolute safe for array iteration
    C,
    /// Column-major order.
    ///
    /// - absolute safe for array iteration
    F,
    /// Automatically choose row/col-major order.
    ///
    /// - try c/f-contig first (also see [`TensorIterOrder::B`]),
    /// - try c/f-prefer second (also see [`TensorIterOrder::C`],
    ///   [`TensorIterOrder::F`]),
    /// - otherwise [`FlagOrder::default()`], which is defined by crate feature
    ///   `f_prefer`.
    ///
    /// - safe for multi-array iteration like `get_iter(a, b)`
    /// - not safe for cases like `a.iter().zip(b.iter())`
    A,
    /// Greedy when possible (reorder layouts during iteration).
    ///
    /// - safe for multi-array iteration like `get_iter(a, b)`
    /// - not safe for cases like `a.iter().zip(b.iter())`
    /// - if it is used to create a new array, the stride of new array will be
    ///   in K order
    K,
    /// Greedy when possible (reset dimension to 1 if axis is broadcasted).
    ///
    /// - not safe for multi-array iteration like `get_iter(a, b)`
    /// - this is useful for inplace-assign broadcasted array.
    G,
    /// Sequential buffer.
    ///
    /// - not safe for multi-array iteration like `get_iter(a, b)`
    /// - this is useful for reshaping or all-contiguous cases.
    B,
}

impl_changeable_default!(TensorIterOrder, DEFAULT_TENSOR_ITER_ORDER, TensorIterOrder::K);

/* #endregion */

/* #region TensorCopyPolicy */

/// The policy of copying tensor.
pub mod TensorCopyPolicy {
    #![allow(non_snake_case)]

    // this is a workaround in stable rust
    // when const enum can not be used as generic parameters

    pub type FlagCopy = u8;

    /// Copy when needed
    pub const COPY_NEEDED: FlagCopy = 0;
    /// Force copy
    pub const COPY_TRUE: FlagCopy = 1;
    /// Force not copy; and when copy is required, it will emit error
    pub const COPY_FALSE: FlagCopy = 2;

    pub const COPY_DEFAULT: FlagCopy = COPY_NEEDED;
}

/* #endregion */

/* #region blas-flags */

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagTrans {
    /// No transpose
    N = 111,
    /// Transpose
    T = 112,
    /// Conjugate transpose
    C = 113,
    // Conjuate only
    CN = 114,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagSide {
    /// Left side
    L = 141,
    /// Right side
    R = 142,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagUpLo {
    /// Upper triangle
    U = 121,
    /// Lower triangle
    L = 122,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagDiag {
    /// Non-unit diagonal
    N = 131,
    /// Unit diagonal
    U = 132,
}

/* #endregion */

/* #region symm-flags */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagSymm {
    /// Symmetric matrix
    Sy,
    /// Hermitian matrix
    He,
    /// Anti-symmetric matrix
    Ay,
    /// Anti-Hermitian matrix
    Ah,
    /// Non-symmetric matrix
    N,
}

pub type TensorOrder = FlagOrder;
pub type TensorDiag = FlagDiag;
pub type TensorSide = FlagSide;
pub type TensorUpLo = FlagUpLo;
pub type TensorTrans = FlagTrans;
pub type TensorSymm = FlagSymm;

/* #endregion */

/* #region flag alias */

pub use FlagTrans::C as ConjTrans;
pub use FlagTrans::N as NoTrans;
pub use FlagTrans::T as Trans;

pub use FlagSide::L as Left;
pub use FlagSide::R as Right;

pub use FlagUpLo::L as Lower;
pub use FlagUpLo::U as Upper;

pub use FlagDiag::N as NonUnit;
pub use FlagDiag::U as Unit;

pub use FlagOrder::C as RowMajor;
pub use FlagOrder::F as ColMajor;

/* #endregion */

/* #region flag into */

impl From<char> for FlagTrans {
    fn from(val: char) -> Self {
        match val {
            'N' | 'n' => FlagTrans::N,
            'T' | 't' => FlagTrans::T,
            'C' | 'c' => FlagTrans::C,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<FlagTrans> for char {
    fn from(val: FlagTrans) -> Self {
        match val {
            FlagTrans::N => 'N',
            FlagTrans::T => 'T',
            FlagTrans::C => 'C',
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<FlagTrans> for c_char {
    fn from(val: FlagTrans) -> Self {
        match val {
            FlagTrans::N => b'N' as c_char,
            FlagTrans::T => b'T' as c_char,
            FlagTrans::C => b'C' as c_char,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<c_char> for FlagTrans {
    fn from(val: c_char) -> Self {
        match val as u8 {
            b'N' => FlagTrans::N,
            b'T' => FlagTrans::T,
            b'C' => FlagTrans::C,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<CBLAS_TRANSPOSE> for FlagTrans {
    fn from(val: CBLAS_TRANSPOSE) -> Self {
        match val {
            CBLAS_TRANSPOSE::CblasNoTrans => FlagTrans::N,
            CBLAS_TRANSPOSE::CblasTrans => FlagTrans::T,
            CBLAS_TRANSPOSE::CblasConjTrans => FlagTrans::C,
        }
    }
}

impl From<FlagTrans> for CBLAS_TRANSPOSE {
    fn from(val: FlagTrans) -> Self {
        match val {
            FlagTrans::N => CBLAS_TRANSPOSE::CblasNoTrans,
            FlagTrans::T => CBLAS_TRANSPOSE::CblasTrans,
            FlagTrans::C => CBLAS_TRANSPOSE::CblasConjTrans,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<char> for FlagDiag {
    fn from(val: char) -> Self {
        match val {
            'N' | 'n' => FlagDiag::N,
            'U' | 'u' => FlagDiag::U,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<FlagDiag> for char {
    fn from(val: FlagDiag) -> Self {
        match val {
            FlagDiag::N => 'N',
            FlagDiag::U => 'U',
        }
    }
}

impl From<FlagDiag> for c_char {
    fn from(val: FlagDiag) -> Self {
        match val {
            FlagDiag::N => b'N' as c_char,
            FlagDiag::U => b'U' as c_char,
        }
    }
}

impl From<c_char> for FlagDiag {
    fn from(val: c_char) -> Self {
        match val as u8 {
            b'N' => FlagDiag::N,
            b'U' => FlagDiag::U,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<CBLAS_DIAG> for FlagDiag {
    fn from(val: CBLAS_DIAG) -> Self {
        match val {
            CBLAS_DIAG::CblasNonUnit => FlagDiag::N,
            CBLAS_DIAG::CblasUnit => FlagDiag::U,
        }
    }
}

impl From<FlagDiag> for CBLAS_DIAG {
    fn from(val: FlagDiag) -> Self {
        match val {
            FlagDiag::N => CBLAS_DIAG::CblasNonUnit,
            FlagDiag::U => CBLAS_DIAG::CblasUnit,
        }
    }
}

impl From<char> for FlagSide {
    fn from(val: char) -> Self {
        match val {
            'L' | 'l' => FlagSide::L,
            'R' | 'r' => FlagSide::R,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<FlagSide> for char {
    fn from(val: FlagSide) -> Self {
        match val {
            FlagSide::L => 'L',
            FlagSide::R => 'R',
        }
    }
}

impl From<FlagSide> for c_char {
    fn from(val: FlagSide) -> Self {
        match val {
            FlagSide::L => b'L' as c_char,
            FlagSide::R => b'R' as c_char,
        }
    }
}

impl From<c_char> for FlagSide {
    fn from(val: c_char) -> Self {
        match val as u8 {
            b'L' => FlagSide::L,
            b'R' => FlagSide::R,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<CBLAS_SIDE> for FlagSide {
    fn from(val: CBLAS_SIDE) -> Self {
        match val {
            CBLAS_SIDE::CblasLeft => FlagSide::L,
            CBLAS_SIDE::CblasRight => FlagSide::R,
        }
    }
}

impl From<FlagSide> for CBLAS_SIDE {
    fn from(val: FlagSide) -> Self {
        match val {
            FlagSide::L => CBLAS_SIDE::CblasLeft,
            FlagSide::R => CBLAS_SIDE::CblasRight,
        }
    }
}

impl From<char> for FlagUpLo {
    fn from(val: char) -> Self {
        match val {
            'U' | 'u' => FlagUpLo::U,
            'L' | 'l' => FlagUpLo::L,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<FlagUpLo> for char {
    fn from(val: FlagUpLo) -> Self {
        match val {
            FlagUpLo::U => 'U',
            FlagUpLo::L => 'L',
        }
    }
}

impl From<FlagUpLo> for c_char {
    fn from(val: FlagUpLo) -> Self {
        match val {
            FlagUpLo::U => b'U' as c_char,
            FlagUpLo::L => b'L' as c_char,
        }
    }
}

impl From<c_char> for FlagUpLo {
    fn from(val: c_char) -> Self {
        match val as u8 {
            b'U' => FlagUpLo::U,
            b'L' => FlagUpLo::L,
            _ => rstsr_invalid!(val).unwrap(),
        }
    }
}

impl From<CBLAS_UPLO> for FlagUpLo {
    fn from(val: CBLAS_UPLO) -> Self {
        match val {
            CBLAS_UPLO::CblasUpper => FlagUpLo::U,
            CBLAS_UPLO::CblasLower => FlagUpLo::L,
        }
    }
}

impl From<FlagUpLo> for CBLAS_UPLO {
    fn from(val: FlagUpLo) -> Self {
        match val {
            FlagUpLo::U => CBLAS_UPLO::CblasUpper,
            FlagUpLo::L => CBLAS_UPLO::CblasLower,
        }
    }
}

impl From<CBLAS_LAYOUT> for FlagOrder {
    fn from(val: CBLAS_LAYOUT) -> Self {
        match val {
            CBLAS_LAYOUT::CblasRowMajor => FlagOrder::C,
            CBLAS_LAYOUT::CblasColMajor => FlagOrder::F,
        }
    }
}

impl From<FlagOrder> for CBLAS_LAYOUT {
    fn from(val: FlagOrder) -> Self {
        match val {
            FlagOrder::C => CBLAS_LAYOUT::CblasRowMajor,
            FlagOrder::F => CBLAS_LAYOUT::CblasColMajor,
        }
    }
}

/* #endregion */

/* #region flag flip */

impl FlagOrder {
    pub fn flip(&self) -> Self {
        match self {
            FlagOrder::C => FlagOrder::F,
            FlagOrder::F => FlagOrder::C,
        }
    }
}

impl FlagTrans {
    pub fn flip(&self, hermi: bool) -> Result<Self> {
        match self {
            FlagTrans::N => match hermi {
                true => Ok(FlagTrans::C),
                false => Ok(FlagTrans::T),
            },
            FlagTrans::T => Ok(FlagTrans::N),
            FlagTrans::C => Ok(FlagTrans::N),
            _ => rstsr_invalid!(self)?,
        }
    }
}

impl FlagSide {
    pub fn flip(&self) -> Result<Self> {
        match self {
            FlagSide::L => Ok(FlagSide::R),
            FlagSide::R => Ok(FlagSide::L),
        }
    }
}

impl FlagUpLo {
    pub fn flip(&self) -> Result<Self> {
        match self {
            FlagUpLo::U => Ok(FlagUpLo::L),
            FlagUpLo::L => Ok(FlagUpLo::U),
        }
    }
}

/* #endregion */
