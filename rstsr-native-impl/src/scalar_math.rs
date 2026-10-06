//! Scalar math helpers shared by the elementwise device kernels.

use num::Float;

/// IEEE roundToIntegralTiesToEven: the standard's `round` (and numpy's
/// `np.round`) resolve halfway cases to the even integer, while Rust's
/// `f64::round` is ties-away. f32 rounds through f64 exactly; NaN/inf and
/// signed zeros propagate per IEEE (`round(-0.5) == -0.0`).
#[cfg(feature = "std")]
pub fn round_ties_even_f<T: Float>(b: T) -> T {
    // ToPrimitive/NumCast are total for f32/f64, so the fallbacks are
    // unreachable; NaN keeps a future failure loud instead of silently
    // returning the unrounded input.
    let x = num::ToPrimitive::to_f64(&b).unwrap_or(f64::NAN);
    num::NumCast::from(x.round_ties_even()).unwrap_or_else(T::nan)
}

/// no_std variant of [`round_ties_even_f`]: pure arithmetic, since
/// `f64::round_ties_even` is std-only.
#[cfg(not(feature = "std"))]
pub fn round_ties_even_f<T: Float>(b: T) -> T {
    let r = b.round();
    let diff = b - r;
    let two = T::one() + T::one();
    if diff == two.recip() || diff == -two.recip() {
        // halfway: r odd -> step to the even neighbor; a zero result
        // takes the sign of b, matching IEEE (round(-0.5) == -0.0)
        let even = if r % two != T::zero() { r - b.signum() } else { r };
        if even == T::zero() && b.is_sign_negative() {
            -even
        } else {
            even
        }
    } else {
        r
    }
}
