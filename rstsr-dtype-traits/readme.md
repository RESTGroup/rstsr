# RSTSR dtype Traits

This trait tries to resolve some function clashes in crate `num`. For example:
- `abs` function may have different behaviors for `Signed` and `ComplexFloat`,
- `max` and `min` function is defined by `f64`, trait `Ord` (where `max` or `min` is defined in standard library) is not implemented for `f64`.

For those functions, it is more proper to define traits for each function specifically.

We only implements several types in standard library, or complex values. More types (from other crates) will be added in future or by issue request.
