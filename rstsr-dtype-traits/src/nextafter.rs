//! This module provides traits and implementations for the nextafter operation.
//!
//! # See also
//!
//! - [Document of NumPy](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html)
//! - [Python Array API](https://data-apis.org/array-api/latest/API_specification/generated/array_api.nextafter.html)

#[cfg(feature = "half")]
use half::{bf16, f16};

pub trait NextAfterAPI {
    /// Returns the next representable floating-point value after `self` in the direction of
    /// `other`.
    fn nextafter(self, other: Self) -> Self;
}

#[cfg(feature = "half")]
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

impl NextAfterAPI for f32 {
    fn nextafter(self, other: Self) -> Self {
        libm::nextafterf(self, other)
    }
}

impl NextAfterAPI for f64 {
    fn nextafter(self, other: Self) -> Self {
        libm::nextafter(self, other)
    }
}

#[cfg(feature = "half")]
impl NextAfterAPI for f16 {
    fn nextafter(self, other: Self) -> Self {
        nextafter_f16(self, other)
    }
}

#[cfg(feature = "half")]
impl NextAfterAPI for bf16 {
    fn nextafter(self, other: Self) -> Self {
        nextafter_bf16(self, other)
    }
}

#[cfg(test)]
#[cfg(feature = "half")]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn test_nextafter_f16() {
        // Test basic functionality
        let x = f16::from_f32(1.0);
        let y = f16::from_f32(2.0);
        let result = x.nextafter(y);
        assert!(result > x);
        assert_eq!(result - x, f16::EPSILON);
        println!("Result: {:?}", result);

        // Test equal values
        let z = f16::from_f32(1.0);
        assert_eq!(x.nextafter(z), z);

        // Test NaN
        assert!(f16::NAN.nextafter(f16::from_f32(1.0)).is_nan());
    }

    #[test]
    fn test_nextafter_bf16() {
        // Test basic functionality
        let x = bf16::from_f32(1.0);
        let y = bf16::from_f32(2.0);
        let result = x.nextafter(y);
        assert!(result > x);
        assert_eq!(result - x, bf16::EPSILON);
        println!("Result: {:?}", result);

        // Test equal values
        let z = bf16::from_f32(1.0);
        assert_eq!(x.nextafter(z), z);

        // Test NaN
        assert!(bf16::NAN.nextafter(bf16::from_f32(1.0)).is_nan());
    }
}
