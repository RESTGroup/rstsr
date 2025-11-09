use crate::*;
use core::ops::*;
use derive_builder::Builder;

/// Arguments for `isclose` function.
///
/// For type [f64], you can also use the `From` trait implementations to create
/// `IsCloseArgs<f64>` from simpler types:
/// - `f64`: specifies `rtol`;
/// - `(f64, f64)`: specifies `rtol` and `atol`;
/// - `(f64, f64, bool)`: specifies `rtol`, `atol`, and `equal_nan`;
/// - `None`: uses default values.
///
/// # See also
///
/// [`isclose`](isclose())
#[derive(Builder, Clone, PartialEq, Eq, Debug)]
pub struct IsCloseArgs<TE: 'static> {
    /// Relative tolerance. For type [f64], the default is `1.0e-5`.
    #[builder(default = "default_rtol()?")]
    pub rtol: TE,
    /// Absolute tolerance. For type [f64], the default is `1.0e-8`.
    #[builder(default = "default_atol()?")]
    pub atol: TE,
    /// Whether to consider NaNs as equal. For type [f64], the default is `false`.
    #[builder(default = "false")]
    pub equal_nan: bool,
}

fn default_rtol<TE: 'static>() -> Result<TE, String> {
    use core::any::*;
    if TypeId::of::<TE>() == TypeId::of::<f64>() {
        Ok(unsafe { core::mem::transmute_copy::<f64, TE>(&1.0e-5_f64) })
    } else {
        let type_name = type_name::<TE>();
        Err(format!("default rtol is not defined for type `{}`. Please specify `rtol` explicitly.", type_name))
    }
}

fn default_atol<TE: 'static>() -> Result<TE, String> {
    use core::any::*;
    if TypeId::of::<TE>() == TypeId::of::<f64>() {
        Ok(unsafe { core::mem::transmute_copy::<f64, TE>(&1.0e-8_f64) })
    } else {
        let type_name = type_name::<TE>();
        Err(format!("default atol is not defined for type `{}`. Please specify `atol` explicitly.", type_name))
    }
}

impl Default for IsCloseArgs<f64> {
    fn default() -> Self {
        Self { rtol: 1.0e-5, atol: 1.0e-8, equal_nan: false }
    }
}

/// Checks whether two numbers are close to each other within a given tolerance.
///
/// # Notes to definition of closeness
///
/// For finite values, isclose uses the following equation to test whether two floating point values
/// are equivalent:
///
/// ```text
/// |a - b| <= atol + rtol * |b|
/// ```
///
/// Note that this equation is not symmetric in `a` and `b`: it assumes that `b` is the reference
/// value; so that `isclose(a, b, args)` may not be the same as `isclose(b, a, args)`.
///
/// # Notes to arguments
///
/// The argument `args` in this function should be usually of type [f64]. You can create it
/// - manually by specifying all fields;
/// - by using the `From` trait implementations for type [`IsCloseArgs<f64>`]:
///   - `f64`: specifies `rtol`;
///   - `(f64, f64)`: specifies `rtol` and `atol`;
///   - `(f64, f64, bool)`: specifies `rtol`, `atol`, and `equal_nan`;
///   - `None`: uses default values.
/// - by using the builder pattern:
///
/// ```rust
/// # use rstsr_dtype_traits::{isclose, IsCloseArgs, IsCloseArgsBuilder};
/// let args_by_builder = IsCloseArgsBuilder::<f64>::default()
///     .rtol(1.0e-6)
///     .atol(1.0e-9)
///     .equal_nan(true)
///     .build().unwrap();
/// let args_by_tuple: IsCloseArgs<f64> = (1.0e-6, 1.0e-9, true).into();
/// assert_eq!(args_by_builder, args_by_tuple);
/// ```
#[inline]
pub fn isclose<TA, TB, TE>(a: &TA, b: &TB, args: &IsCloseArgs<TE>) -> bool
where
    TA: Clone + DTypePromoteAPI<TB>,
    TB: Clone,
    <TA as DTypePromoteAPI<TB>>::Res: ExtNum<AbsOut: DTypeCastAPI<TE>>,
    TE: ExtFloat + Add<TE, Output = TE> + Mul<TE, Output = TE> + PartialOrd + Clone,
{
    let IsCloseArgs { rtol, atol, equal_nan } = args;
    let (a, b) = TA::promote_pair(a.clone(), b.clone());
    let diff: TE = a.clone().ext_abs_diff(b.clone()).into_cast();
    let abs_b: TE = b.clone().ext_abs().into_cast();
    let comp = diff <= atol.clone() + rtol.clone() * abs_b;
    let nan_check = *equal_nan && a.is_nan() && b.is_nan();
    comp || nan_check
}

impl From<f64> for IsCloseArgs<f64> {
    #[inline]
    fn from(rtol: f64) -> Self {
        Self { rtol, atol: 1.0e-8, equal_nan: false }
    }
}

impl From<(f64,)> for IsCloseArgs<f64> {
    #[inline]
    fn from(v: (f64,)) -> Self {
        let (rtol,) = v;
        Self { rtol, atol: 1.0e-8, equal_nan: false }
    }
}

impl From<(f64, f64)> for IsCloseArgs<f64> {
    #[inline]
    fn from(v: (f64, f64)) -> Self {
        let (rtol, atol) = v;
        Self { rtol, atol, equal_nan: false }
    }
}

impl From<(f64, f64, bool)> for IsCloseArgs<f64> {
    #[inline]
    fn from(v: (f64, f64, bool)) -> Self {
        let (rtol, atol, equal_nan) = v;
        Self { rtol, atol, equal_nan }
    }
}

impl From<Option<f64>> for IsCloseArgs<f64> {
    #[inline]
    fn from(rtol: Option<f64>) -> Self {
        match rtol {
            Some(rtol) => Self { rtol, atol: 1.0e-8, equal_nan: false },
            None => Self { rtol: 1.0e-5, atol: 1.0e-8, equal_nan: false },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_isclose_f64() {
        let a = 1.00001_f64;
        let b = 1.00002_f64;
        let args = None.into();
        assert!(isclose(&a, &b, &args));
        let args = IsCloseArgsBuilder::default().rtol(1.0e-6).atol(1.0e-9).equal_nan(false).build().unwrap();
        assert!(!isclose(&a, &b, &args));
    }

    #[test]
    fn test_isclose_usize() {
        let a: usize = 100;
        let b: usize = 102;
        let args = None.into();
        assert!(!isclose(&a, &b, &args));
    }

    #[test]
    fn test_isclose_usize_c32() {
        use num::Complex;
        let a: usize = 100;
        let b: Complex<f32> = Complex::new(100.0, 0.0);
        let args = None.into();
        assert!(isclose(&a, &b, &args));
        let c: Complex<f32> = Complex::new(100.01, 0.0);
        assert!(!isclose(&a, &c, &args));
    }
}
