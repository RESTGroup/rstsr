//! Flags for the crate.

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

/* #region TensorOrder */

/// The order of the tensor.
///
/// # Default
///
/// Default order depends on cargo feature `f_prefer`.
/// If `f_prefer` is set, then [`TensorOrder::F`] is applied as default;
/// otherwise [`TensorOrder::C`] is applied as default.
///
/// # IMPORTANT NOTE
///
/// F-prefer is not a stable feature currently! We develop only in C-prefer
/// currently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOrder {
    /// row-major order.
    C,
    /// column-major order.
    F,
}

#[allow(clippy::derivable_impls)]
impl Default for TensorOrder {
    fn default() -> Self {
        #[cfg(not(feature = "f_prefer"))]
        {
            TensorOrder::C
        }
        #[cfg(feature = "f_prefer")]
        {
            TensorOrder::F
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
    /// - otherwise [`TensorOrder::default()`], which is defined by crate
    ///   feature `f_prefer`.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagTrans {
    #[default]
    Undefined,
    /// No transpose
    N,
    /// Transpose
    T,
    /// Conjugate transpose
    C,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagSide {
    #[default]
    Undefined,
    /// Left side
    L,
    /// Right side
    R,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagUpLo {
    #[default]
    Undefined,
    /// Upper triangle
    U,
    /// Lower triangle
    L,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagDiag {
    #[default]
    Undefined,
    /// Non-unit diagonal
    N,
    /// Unit diagonal
    U,
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

pub type TensorDiag = FlagDiag;
pub type TensorSide = FlagSide;
pub type TensorUpLo = FlagUpLo;
pub type TensorTrans = FlagTrans;
pub type TensorSymm = FlagSymm;

/* #endregion */
