/// Extension trait for float types ([`num::Float`]).
pub trait ExtFloat: Clone {
    /// Returns the next representable floating-point value after `self` in the direction of
    /// `other`.
    fn ext_nextafter(self, other: Self) -> Self;
}

impl ExtFloat for f32 {
    fn ext_nextafter(self, other: Self) -> Self {
        libm::nextafterf(self, other)
    }
}

impl ExtFloat for f64 {
    fn ext_nextafter(self, other: Self) -> Self {
        libm::nextafter(self, other)
    }
}

impl ExtFloat for half::bf16 {
    fn ext_nextafter(self, other: Self) -> Self {
        nextafter_bf16(self, other)
    }
}

impl ExtFloat for half::f16 {
    fn ext_nextafter(self, other: Self) -> Self {
        nextafter_f16(self, other)
    }
}

/* #region implementaion detail */

#[cfg(feature = "half")]
mod impl_half {
    use half::{bf16, f16};

    macro_rules! force_eval {
        ($e:expr) => {
            unsafe { ::core::ptr::read_volatile(&$e) }
        };
    }

    #[cfg(feature = "half")]
    #[inline]
    pub fn nextafter_f16(x: f16, y: f16) -> f16 {
        if x.is_nan() || y.is_nan() {
            return x + y;
        }

        let mut ux_i = x.to_bits();
        let uy_i = y.to_bits();
        if ux_i == uy_i {
            return y;
        }

        let ax = ux_i & 0x7FFF_u16; // Mask for f16 (11 bits exponent + mantissa)
        let ay = uy_i & 0x7FFF_u16;

        if ax == 0 {
            if ay == 0 {
                return y;
            }
            ux_i = (uy_i & 0x8000_u16) | 1; // Smallest subnormal with sign of y
        } else if ax > ay || ((ux_i ^ uy_i) & 0x8000_u16) != 0 {
            ux_i -= 1;
        } else {
            ux_i += 1;
        }

        let e = ux_i & 0x7C00_u16; // Exponent mask for f16 (5 exponent bits)
                                   // raise overflow if ux_f is infinite and x is finite
        if e == 0x7C00_u16 {
            force_eval!(x + x);
        }
        let ux_f = f16::from_bits(ux_i);
        // raise underflow if ux_f is subnormal or zero
        if e == 0 {
            force_eval!(x * x + ux_f * ux_f);
        }
        ux_f
    }

    #[cfg(feature = "half")]
    #[inline]
    pub fn nextafter_bf16(x: bf16, y: bf16) -> bf16 {
        if x.is_nan() || y.is_nan() {
            return x + y;
        }

        let mut ux_i = x.to_bits();
        let uy_i = y.to_bits();
        if ux_i == uy_i {
            return y;
        }

        let ax = ux_i & 0x7FFF_u16; // Mask for bf16 (8 bits exponent + mantissa)
        let ay = uy_i & 0x7FFF_u16;

        if ax == 0 {
            if ay == 0 {
                return y;
            }
            ux_i = (uy_i & 0x8000_u16) | 1; // Smallest subnormal with sign of y
        } else if ax > ay || ((ux_i ^ uy_i) & 0x8000_u16) != 0 {
            ux_i -= 1;
        } else {
            ux_i += 1;
        }

        let e = ux_i & 0x7F80_u16; // Exponent mask for bf16 (8 exponent bits)
                                   // raise overflow if ux_f is infinite and x is finite
        if e == 0x7F80_u16 {
            force_eval!(x + x);
        }
        let ux_f = bf16::from_bits(ux_i);
        // raise underflow if ux_f is subnormal or zero
        if e == 0 {
            force_eval!(x * x + ux_f * ux_f);
        }
        ux_f
    }
}

#[cfg(feature = "half")]
pub use impl_half::*;

/* #endregion */
