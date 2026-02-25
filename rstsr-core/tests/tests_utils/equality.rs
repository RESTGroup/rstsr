use std::fmt::Debug;

use rstsr::prelude::*;
use rstsr_core::storage::OpAllCloseAPI;
use rstsr_dtype_traits::IsCloseArgs;

/// Raises an AssertionError if two objects are not equal.
///
/// This function is similar to `np.testing.assert_equal`, but uses `rt::allclose` for value
/// comparison. This function does not have the same behavior on `NaN` values. Please fill
/// `isclose_arg = IsCloseArgs { equal_nan: true, ..Default::default() }` to make `NaN` values
/// compare equal.
pub fn assert_equal<TA, TB, B, DA, DB>(
    a: impl TensorViewAPI<Type = TA, Backend = B, Dim = DA>,
    b: impl TensorViewAPI<Type = TB, Backend = B, Dim = DB>,
    isclose_arg: impl Into<IsCloseArgs<f64>>,
) where
    TA: Clone + Debug,
    TB: Clone + Debug,
    B: DeviceAPI<TA, Raw: Clone> + DeviceAPI<TB, Raw: Clone> + OpAllCloseAPI<TA, TB, f64, IxDyn> + Debug,
    DA: DimAPI,
    DB: DimAPI,
{
    let (a, b) = (a.view().into_dim::<IxDyn>(), b.view().into_dim::<IxDyn>());
    assert_eq!(a.shape(), b.shape(), "Shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    assert!(rt::allclose(&a, &b, isclose_arg.into()), "Value mismatch: {:?} vs {:?}", a, b);
}
