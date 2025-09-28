use duplicate::duplicate_item;
use num::{complex::ComplexFloat, Complex};

// Extension trait for numerical ([`num::Num`]) types.
pub trait ExtNum: Clone {
    /* #region abs */

    /// The output type of the absolute value operation.
    type AbsOut: Clone + ExtNum<AbsOut = Self::AbsOut>;

    /// Whether the absolute value operation does not change the value.
    const ABS_UNCHANGED: bool;

    /// Whether the output type of the absolute value operation is the same as the input type.
    const ABS_SAME_TYPE: bool;

    /// Computes the absolute value of the number.
    fn ext_abs(self) -> Self::AbsOut;

    /* #endregion */

    /* #region real-imag */

    /// Returns the real part of the number.
    fn ext_real(self) -> Self::AbsOut;

    /// Returns the imaginary part of the number.
    fn ext_imag(self) -> Self::AbsOut;

    /* #endregion */
}

#[duplicate_item(T; [u8]; [u16]; [u32]; [u64]; [u128]; [usize];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = true;
    const ABS_SAME_TYPE: bool = true;
    fn ext_abs(self) -> Self {
        self
    }
    /* #endregion */

    /* #region real-imag */
    fn ext_real(self) -> Self {
        self
    }
    fn ext_imag(self) -> Self {
        0 as Self
    }
}

#[duplicate_item(T; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    fn ext_abs(self) -> Self {
        self.abs()
    }
    /* #endregion */

    /* #region real-imag */
    fn ext_real(self) -> Self {
        self
    }
    fn ext_imag(self) -> Self {
        0 as Self
    }
    /* #endregion */
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    fn ext_abs(self) -> Self {
        self.abs()
    }
    /* #endregion */

    /* #region real-imag */
    fn ext_real(self) -> Self {
        self
    }
    fn ext_imag(self) -> Self {
        0 as Self
    }
    /* #endregion */
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = Self;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = true;
    fn ext_abs(self) -> Self {
        self.abs()
    }
    /* #endregion */

    /* #region real-imag */
    fn ext_real(self) -> Self {
        self
    }
    fn ext_imag(self) -> Self {
        Self::ZERO
    }
    /* #endregion */
}

#[duplicate_item(T; [Complex<f32>]; [Complex<f64>];)]
impl ExtNum for T {
    /* #region abs */
    type AbsOut = <T as ComplexFloat>::Real;
    const ABS_UNCHANGED: bool = false;
    const ABS_SAME_TYPE: bool = false;
    fn ext_abs(self) -> Self::AbsOut {
        self.norm()
    }
    /* #endregion */

    /* #region real-imag */
    fn ext_real(self) -> Self::AbsOut {
        self.re
    }
    fn ext_imag(self) -> Self::AbsOut {
        self.im
    }
    /* #endregion */
}
