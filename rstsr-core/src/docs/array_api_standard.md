# Array API standard Fullfillment

For column status:
- Y: Implemented in rstsr (may not be fully functional like numpy or Python array API specification, but should be enough);
- C: Changed feature (**breaking using experience from numpy**);
- P: Partial implemented in rstsr (not all features in Python array API is implemented);
- D: Features that would be dropped in rstsr.
- blank: Will be implemented in future version of RSTSR.

## Operators

### Arithmetic Operators

| status | implementation | Python API | description |
|-|-|-|-|
| D | | `__pos__` | `+x` |
| Y | `-` | `__neg__` | `-x` |
| Y | `+` | `__add__` | `x1 + x2` |
| Y | `-` | `__sub__` | `x1 - x2` |
| Y | `*` | `__mul__` | `x1 * x2` |
| Y | `/` | `__truediv__` | `x1 / x2` |
| Y | `floor_divide` | `__floordiv__` | `x1 // x2` |
| **C** | [`rt::rem`][^2] | `__mod__` | `x1 % x2` |
| Y | [`pow`] | `__pow__` | `x1 ** x2` |
| Y | `+=` | `__iadd__` | `x1 += x2` |
| Y | `-=` | `__isub__` | `x1 -= x2` |
| Y | `*=` | `__imul__` | `x1 *= x2` |
| Y | `/=` | `__itruediv__` | `x1 /= x2` |
| D | | `__ifloordiv__` | `x1 //= x2` |
| D | | `__ipow__` | `x1 **= x2` |
| Y | `%=` | `__imod__` | `x1 %= x2` |

[^2]: To use remainder (modular) function correctly, one may use `rt::rem` (as function), instead of using `x1 % x2` (as operator) or `x1.rem(x2)` (as associated method). In this crate, the latter two will call [`matmul()`].

**Changed feature**
- `__mod__`: We do not use remainder function to represent something like `8 % 3 = 2`, but instead using notation `%` to represent matrix multiplication (`@` in python/numpy).


**Dropped support**
- `__pos__`: In rust, leading `+` is not allowed.
- `__ifloordiv__`: This is not a priority for implementation.
- `__ipow__`: This is not a priority for implementation.

### Array Operators

| status | implementation | Python API | description |
|-|-|-|-|
| **C** | `%`, [`matmul()`] | `__matmul__` | `x1 @ x2` |
| D | | `__imatmul__` | `x1 @= x2` |

**Changed feature**
- `__matmul__`: In rust, there was discussions whether to implement `@` as matrix multiplication (or other operator notations, since `@` has been used in binary operation for pattern matching). Instead we use notation `%` to represent matrix multiplication (`@` in python/numpy). See `__rem__` function for more information.

Dropped support
- `__imatmul__`: Inplace matmul is not convenient to be realized.

### Bitwise Operators

| status | implementation | Python API | description |
|-|-|-|-|
| Y | `!` [`Not`] | `__invert__` | `~x` |
| Y | `&` [`BitAnd`] | `__and__` | `x1 & x2` |
| Y | `\|` [`BitOr`] | `__or__` | `x1 \| x2` |
| Y | `^`  [`BitXor`] | `__xor__` | `x1 ^ x2` |
| Y | `<<` [`Shl`] | `__lshift__` | `x1 << x2` |
| Y | `>>` [`Shr`] | `__rshift__` | `x1 >> x2` |
| Y | `&=` [`BitAndAssign`] | `__iand__` | `x1 &= x2` |
| Y | `\|=` [`BitOrAssign`] | `__ior__` | `x1 \|= x2` |
| Y | `^=` [`BitXorAssign`] | `__ixor__` | `x1 ^= x2` |
| Y | `<<=` [`ShlAssign`] | `__ilshift__` | `x1 <<= x2` |
| Y | `>>=` [`ShrAssign`] | `__irshift__` | `x1 >>= x2` |

### Comparsion Operators

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`lt`], [`less`] | `__lt__` | `x1 < x2` |
| Y | [`le`], [`less_equal`] | `__le__` | `x1 <= x2` |
| Y | [`gt`], [`greater`] | `__gt__` | `x1 > x2` |
| Y | [`ge`], [`greater_equal`] | `__ge__` | `x1 >= x2` |
| Y | [`eq`], [`equal`] | `__eq__` | `x1 == x2` |
| Y | [`ne`], [`not_equal`] | `__ne__` | `x1 != x2` |

### Array Object Attributes

| status | implementation | Python API | description |
|-|-|-|-|
| C | `T` of [`TensorAny<R, T, B, D>`] | `dtype` | Data type of the array elements. |
| C | `B` of [`TensorAny<R, T, B, D>`] | `device` | Hardware device the array data resides on. |
| Y | [`TensorBase::swapaxes`]`(-1, -2)` | `mT` | Transpose of a matrix (or a stack of matrices). |
| Y | [`TensorBase::ndim`] | `ndim` | Number of array dimensions (axes). |
| Y | [`TensorBase::shape`] | `shape` | Array dimensions. |
| Y | [`TensorBase::size`] | `size` | Number of elements in an array. |
| Y | [`transpose`], [`TensorBase::t`] | `T` | Transpose of the array. |

### Methods

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`abs`] | `__abs__` | Calculates the absolute value for each element of an array instance. |
| Y | [`asarray`] | `__bool__` | Converts a zero-dimensional array to a Python bool object. |
| Y | [`asarray`] | `__complex__` | Converts a zero-dimensional array to a Python `complex` object. |
| Y | [`asarray`] | `__float__` | Converts a zero-dimensional array to a Python `float` object. |
| Y | [`Index`] | `__getitem__` | Returns `self[key]`. |
| Y | [`TensorBase::to_scalar`] | `__index__` | Converts a zero-dimensional integer array to a Python `int` object. |
| Y | [`asarray`] | `__int__` | Converts a zero-dimensional array to a Python `int` object. |
| Y | [`IndexMut`] | `__setitem__` | Sets `self[key]` to `value`. |
| P | [`DeviceChangeAPI::to_device`] | `to_device` | Copy the array from the device on which it currently resides to the specified `device`. |

## Constants

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`core::f64::consts::E`] | [`e`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.constants.e.html) | IEEE 754 floating-point representation of Euler's constant. |
| Y | [`f64::INFINITY`] | [`inf`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.constants.inf.html) | IEEE 754 floating-point representation of (positive) infinity. |
| Y | [`f64::NAN`] | [`nan`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.constants.nan.html) | IEEE 754 floating-point representation of Not a Number (NaN). |
| Y | [`Indexer::Insert`] | [`newaxis`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.constants.newaxis.html) | An alias for None which is useful for indexing arrays. |
| Y | [`core::f64::consts::PI`] | [`pi`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.constants.pi.html) | IEEE 754 floating-point representation of the mathematical constant π. |

## Creation Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`arange`] | [`arange`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.empty.html) | Returns evenly spaced values within the half-open interval `[start, stop)` as a one-dimensional array. |
| P | [`asarray`] | [`asarray`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.asarray.html) | Convert the input to an array. |
| Y | [`empty`] | [`empty`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.empty.html) | Returns an uninitialized array having a specified `shape`. |
| Y | [`empty_like`] | [`empty_like`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.empty_like.html) | Returns an uninitialized array with the same `shape` as an input array `x`. |
| Y | [`eye`] | [`eye`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.eye.html) | Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere. |
| Y | [`full`] | [`full`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.full.html) | Returns a new array having a specified `shape` and filled with `fill_value`. |
| Y | [`full_like`] | [`full_like`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.full_like.html) | Returns a new array filled with fill_value and having the same `shape` as an input array `x`. |
| Y | [`linspace`] | [`linspace`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.linspace.html) | Returns evenly spaced numbers over a specified interval. |
| Y | [`meshgrid`] | `meshgrid` | Returns coordinate matrices from coordinate vectors. |
| Y | [`ones`] | [`ones`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.ones.html) | Returns a new array having a specified shape and filled with ones. |
| Y | [`ones_like`] | [`ones_like`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.ones_like.html) | Returns a new array filled with ones and having the same `shape` as an input array `x`. |
| Y | [`tril`] | `tril` | Returns the lower triangular part of a matrix (or a stack of matrices) `x`. |
| Y | [`triu`] | `triu` | Returns the upper triangular part of a matrix (or a stack of matrices) `x`. |
| Y | [`zeros`] | [`zeros`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.zeros.html) | Returns a new array having a specified `shape` and filled with zeros. |
| Y | [`zeros_like`] | [`zeros_like`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.zeros_like.html) | Returns a new array filled with zeros and having the same `shape` as an input array x. |

**Partial implementation**
- [`asarray`]: This function have different implementations for `Vec<T>`, `[T; N]` and [`Tensor<T, B, D>`]. Different signatures are utilized for different inputs and purposes.

## Data Type

### Data Type Functions

Data type promotion rules are handled by devices instead of core functions (you can implement your promotion rule for your device).

The reference implementation (as in [`DeviceCpuSerial`] and [`DeviceFaer`]), follows two kinds of promotion rules:
- The rule from operator itself: this affects most functions that comes with intrinsic operator (+, -, *, /, &, |, ...), as well as some functions ([`abs`], [`real`], [`imag`], [`pow`], etc.); also see crate [`num`].
- The rule from promotion rule (the same rule of [NumPy's](https://numpy.org/doc/stable/reference/arrays.promotion.html)): this affects most common functions ([`sin`], [`greater`], [`sqrt`], to list a few).

| status | implementation | Python API | description |
|-|-|-|-|
| T | Rust type cast [`as`](https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions)<br>[`DTypePromotionAPI::promote_astype`]<br>[`num::ToPrimitive`] | `astype` | Copies an array to a specified data type irrespective of Type Promotion Rules rules. |
| T | [`DTypePromotionAPI::CAN_CAST_SELF`]<br>[`DTypePromotionAPI::CAN_CAST_OTHER`]<br>operator trait definition | `can_cast` | Determines if one data type can be cast to another data type according Type Promotion Rules rules. |
| T | [`num::Float`] | `finfo` | Machine limits for floating-point data types. |
| T | [`num::Integer`] | `iinfo` | Machine limits for integer data types. |
| T | [`core::any::TypeId`] | `isdtype` | Returns a boolean indicating whether a provided dtype is of a specified data type "kind". |
| T | [`DTypePromotionAPI::Res`] | `result_type` | Returns the dtype that results from applying the type promotion rules (see Type Promotion Rules) to the arguments. |

### Data Type Categories

| rust trait or struct | data type category | dtypes |
|-|-|-|
| [`Num`], [`ExtNum`] | Numeric | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128 |
| [`ExtReal`] | Real-valued | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64 |
| [`Integer`] | Integer | int8, int16, int32, int64, uint8, uint16, uint32, uint64 |
| [`ComplexFloat`] | Floating-point | float32, float64, complex64, complex128 |
| [`Float`], [`ExtFloat`] | Real-valued floating-point | float32, float64 |
| [`Complex`] | Complex floating-point | complex64, complex128 |
| [`bool`] | Boolean | bool |


## Element-wise Functions

### Unary Functions

| status | implementation | Python API | description |
|-|-|-|-|
| [`ExtNum`] | [`abs`] | `abs` | Calculates the absolute value for each element x_i of the input array x. |
| [`ComplexFloat`] | [`acos`] | `acos` | Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element x_i of the input array x. |
| [`ComplexFloat`] | [`acosh`] | `acosh` | Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element x_i of the input array x. |
| [`ComplexFloat`] | [`asin`] | `asin` | Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element x_i of the input array x. |
| [`ComplexFloat`] | [`asinh`] | `asinh` | Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element x_i in the input array x. |
| [`ComplexFloat`] | [`atan`] | `atan` | Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element x_i of the input array x. |
| [`ComplexFloat`] | [`atanh`] | `atanh` | Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element x_i of the input array x. |
| [`Not`] | `!`, [`not`] | `bitwise_invert` | Inverts (flips) each bit for each element x_i of the input array x. |
| [`Float`] | [`ceil`] | `ceil` | Rounds each element x_i of the input array x to the smallest (i.e., closest to -infinity) integer-valued number that is not less than x_i. |
| [`ComplexFloat`] | [`conj`] | `conj` | Returns the complex conjugate for each element x_i of the input array x. |
| [`ComplexFloat`] | [`cos`] | `cos` | Calculates an implementation-dependent approximation to the cosine for each element x_i of the input array x. |
| [`ComplexFloat`] | [`cosh`] | `cosh` | Calculates an implementation-dependent approximation to the hyperbolic cosine for each element x_i in the input array x. |
| [`ComplexFloat`] | [`exp`] | `exp` | Calculates an implementation-dependent approximation to the exponential function for each element x_i of the input array x (e raised to the power of x_i, where e is the base of the natural logarithm). |
| [`Float`] (partial) | [`expm1`] | `expm1` | Calculates an implementation-dependent approximation to exp(x)-1 for each element x_i of the input array x. |
| [`Float`] | [`floor`] | `floor` | Rounds each element x_i of the input array x to the greatest (i.e., closest to +infinity) integer-valued number that is not greater than x_i. |
| [`ExtNum`] | [`imag`] | `imag` | Returns the imaginary component of a complex number for each element x_i of the input array x. |
| [`ComplexFloat`] | [`is_finite`] | `isfinite` | Tests each element x_i of the input array x to determine if finite. |
| [`ComplexFloat`] | [`is_inf`] | `isinf` | Tests each element x_i of the input array x to determine if equal to positive or negative infinity. |
| [`ComplexFloat`] | [`is_nan`] | `isnan` | Tests each element x_i of the input array x to determine whether the element is NaN. |
| [`ComplexFloat`] | [`log`] | `log` | Calculates an implementation-dependent approximation to the natural (base e) logarithm for each element x_i of the input array x. |
| Not implemented | | `log1p` | Calculates an implementation-dependent approximation to log(1+x), where log refers to the natural (base e) logarithm, for each element x_i of the input array x. |
| [`ComplexFloat`] | [`log2`] | `log2` | Calculates an implementation-dependent approximation to the base 2 logarithm for each element x_i of the input array x. |
| [`ComplexFloat`] | [`log10`] | `log10` | Calculates an implementation-dependent approximation to the base 10 logarithm for each element x_i of the input array x. |
| [`Not`] | [`not`] instead | `logical_not` | Computes the logical NOT for each element x_i of the input array x. |
| [`Neg`] | `-`, [`neg`] | `negative` | Computes the numerical negative of each element x_i (i.e., y_i = -x_i) of the input array x. |
| Dropped | | `positive` | Computes the numerical positive of each element x_i (i.e., y_i = +x_i) of the input array x. |
| [`ExtNum`] | [`real`] | `real` | Returns the real component of a complex number for each element x_i of the input array x. |
| [`ComplexFloat`] | [`reciprocal`] | `reciprocal` | Returns the reciprocal for each element x_i of the input array x. |
| [`Float`] | [`round`] | `round` | Rounds each element x_i of the input array x to the nearest integer-valued number. |
| [`ComplexFloat`] | [`sign`] | `sign` | Returns an indication of the sign of a number for each element x_i of the input array x. |
| [`Signed`] | [`signbit`] | `signbit` | Determines whether the sign bit is set for each element x_i of the input array x. |
| [`ComplexFloat`] | [`sin`] | `sin` | Calculates an implementation-dependent approximation to the sine for each element x_i of the input array x. |
| [`ComplexFloat`] | [`sinh`] | `sinh` | Calculates an implementation-dependent approximation to the hyperbolic sine for each element x_i of the input array x. |
| [`Num`] | [`square`] | `square` | Squares each element x_i of the input array x. |
| [`ComplexFloat`] | [`sqrt`] | `sqrt` | Calculates the principal square root for each element x_i of the input array x. |
| [`ComplexFloat`] | [`tan`] | `tan` | Calculates an implementation-dependent approximation to the tangent for each element x_i of the input array x. |
| [`ComplexFloat`] | [`tanh`] | `tanh` | Calculates an implementation-dependent approximation to the hyperbolic tangent for each element x_i of the input array x. |
| [`Float`] | [`trunc`] | `trunc` | Rounds each element x_i of the input array x to the nearest integer-valued number that is closer to zero than x_i. |

### Binary Functions

| status | implementation | Python API | description |
|-|-|-|-|
| [`Add`] | `+`, [`add`] | `add` | Calculates the sum for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Float`] | [`atan2`] | `atan2` | Calculates an implementation-dependent approximation of the inverse tangent of the quotient x1/x2, having domain [-infinity, +infinity] x [-infinity, +infinity] (where the x notation denotes the set of ordered pairs of elements (x1_i, x2_i)) and codomain [-π, +π], for each pair of elements (x1_i, x2_i) of the input arrays x1 and x2, respectively. |
| [`BitAnd`] | `&`, [`bitand`] | `bitwise_and` | Computes the bitwise AND of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Shl`] | `<<`, [`shl`] | `bitwise_left_shift` | Shifts the bits of each element x1_i of the input array x1 to the left by appending x2_i (i.e., the respective element in the input array x2) zeros to the right of x1_i. |
| [`BitOr`] | `\|`, [`bitor`] | `bitwise_or` | Computes the bitwise OR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Shr`] | `>>`, [`shr`] | `bitwise_right_shift` | Shifts the bits of each element x1_i of the input array x1 to the right according to the respective element x2_i of the input array x2. |
| [`BitXor`] | `^`, [`bitxor`] | `bitwise_xor` | Computes the bitwise XOR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Float`] | [`copysign`] | `copysign` | Composes a floating-point value with the magnitude of x1_i and the sign of x2_i for each element of the input array x1. |
| [`Div`] | `/`, [`div`] | `divide` | Calculates the division of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`PartialEq`] | [`eq`], [`equal`] | `equal` | Computes the truth value of x1_i == x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`ExtReal`] | [`floor_divide`] | `floor_divide` | Rounds the result of dividing each element x1_i of the input array x1 by the respective element x2_i of the input array x2 to the greatest (i.e., closest to +infinity) integer-value number that is not greater than the division result. |
| [`PartialOrd`] | [`gt`], [`greater`] | `greater` | Computes the truth value of x1_i > x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`PartialOrd`] | [`ge`], [`greater_equal`] | `greater_equal` | Computes the truth value of x1_i >= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Float`] | [`hypot`] | `hypot` | Computes the square root of the sum of squares for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`PartialOrd`] | [`lt`], [`less`] | `less` | Computes the truth value of x1_i < x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`PartialOrd`] | [`le`], [`less_equal`] | `less_equal` | Computes the truth value of x1_i <= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`ComplexFloat`] | [`log_add_exp`] | `logaddexp` | Calculates the logarithm of the sum of exponentiations log(exp(x1) + exp(x2)) for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | [`bitand`] instead | `logical_and` | Computes the logical AND for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | [`bitor`] instead | `logical_or` | Computes the logical OR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | [`bitxor`] instead | `logical_xor` | Computes the logical XOR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`ExtReal`] | [`maximum`] | `maximum` | Computes the maximum value for each element x1_i of the input array x1 relative to the respective element x2_i of the input array x2. |
| [`ExtReal`] | [`minimum`] | `minimum` | Computes the minimum value for each element x1_i of the input array x1 relative to the respective element x2_i of the input array x2. |
| [`Mul`] | [`mul`] | `multiply` | Calculates the product for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`ExtFloat`] | [`nextafter`] | `nextafter` | Returns the next representable floating-point value for each element x1_i of the input array x1 in the direction of the respective element x2_i of the input array x2. |
| [`PartialEq`] | [`ne`], [`not_equal`] | `not_equal` | Computes the truth value of x1_i != x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| [`Pow`] | [`pow`] | `pow` | Calculates an implementation-dependent approximation of exponentiation by raising each element x1_i (the base) of the input array x1 to the power of x2_i (the exponent), where x2_i is the corresponding element of the input array x2. |
| [`Rem`] | [`rt::rem`][^2] | `remainder` | Returns the remainder of division for each element x1_i of the input array x1 and the respective element x2_i of the input array x2. |
| [`Sub`] | `-`, [`sub`] | `subtract` | Calculates the difference for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |

### Other functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `clip` | Clamps each element x_i of the input array x to the range [min, max]. |

## Indexing Functions

| status | implementation | Python API | description |
|-|-|-|-|
| P | [`take`] | `take` | Returns elements of an array along an axis. |
| | | `take_along_axis` | Returns elements from an array at the one-dimensional indices specified by indices along a provided axis. |

**Partial implementation**
- [`take`] currently only supports indexing from an axis, which is also the Python Array API requires. However, NumPy also allows `axis = None` to index the flattened array, which is not implemented in RSTSR.

## Inspection

| status | implementation | Python API | description |
|-|-|-|-|
| D | requires reflection | `capabilities` | Returns a dictionary of array library capabilities. |
| Fixed | [`DeviceCpu`] | `default_device` | Returns the default device. |
| D | controled by rust | `default_dtypes` | Returns a dictionary containing default data types. |
| Y | [`TensorBase::device`] | `devices` | Returns a list of supported devices which are available at runtime. |
| Y | [`core::any::type_name_of_val`] | `dtypes` | Returns a dictionary of supported Array API data types. |

## Linear Algebra Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`matmul()`] | `matmul` | Computes the matrix product. |
| Y | [`swapaxes`]`(-1, -2)` | `matrix_transpose` | Transposes a matrix (or a stack of matrices) x. |
| | | `tensordot` | Returns a tensor contraction of x1 and x2 over specific axes. |
| | | `vecdot` | Computes the (vector) dot product of two arrays. |

## Manipulation Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`broadcast_arrays`] | [`broadcast_arrays`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_arrays.html) | Broadcasts one or more arrays against one another. |
| Y | [`to_broadcast`] | [`broadcast_to`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.broadcast_to.html) | Broadcasts an array to a specified shape. |
| Y | [`concat`](concat()) | `concat` | Joins a sequence of arrays along an existing axis. |
| Y | [`expand_dims`] | [`expand_dims`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.expand_dims.html) | Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by `axis`. |
| Y | [`flip`] | [`flip`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.flip.html) | Reverses the order of elements in an array along the given axis. |
| | | `moveaxis` | Moves array axes (dimensions) to new positions, while leaving other axes in their original positions. |
| Y | [`transpose`], [`permute_dims`] | [`permute_dims`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.permute_dims.html) | Permutes the axes (dimensions) of an array `x`. |
| | | `repeat` | Repeats each element of an array a specified number of times on a per-element basis. |
| P | [`reshape`], [`into_shape_assume_contig`] | [`reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html) | Reshapes an array without changing its data. |
| | | `roll` | Rolls array elements along a specified axis. |
| P | [`squeeze`] | [`squeeze`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.squeeze.html) | Removes singleton dimensions (axes) from x. |
| Y | [`stack`] | `stack` | Joins a sequence of arrays along a new axis. |
| | | `tile` | Constructs an array by tiling an input array. |
| Y | [`unstack`] | `unstack` | Splits an array into a sequence of arrays along the given axis. |

**Partial implementation**
- [`squeeze`] accepts one axis as input, instead of accepting multiple axes. This is mostly because output of smaller dimension tensor can be fixed-dimension array ([`DimSmallerOneAPI::SmallerOne`]) when only one axis is passed as argument.
- [`reshape`]: Currently reshape is work-in-progress. It does not copy array when c-contiguous. For numpy, much more cases may not invoke explicit copy when reshape.

## Searching Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`argmax`], [`argmax_axes`] | `argmax` | Returns the indices of the maximum values along a specified axis. |
| Y | [`argmin`], [`argmin_axes`] | `argmin` | Returns the indices of the minimum values along a specified axis. |
| Y | [`count_nonzero`], [`count_nonzero_axes`] | `count_nonzero` | Counts the number of array elements which are non-zero. |
| | | `nonzero` | Returns the indices of the array elements which are non-zero. |
| | | `searchsorted` | Finds the indices into x1 such that, if the corresponding elements in x2 were inserted before the indices, the order of x1, when sorted in ascending order, would be preserved. |
| | | `where` | Returns elements chosen from x1 or x2 depending on condition. |

## Set Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `unique_all` | Returns the unique elements of an input array x, the first occurring indices for each unique element in x, the indices from the set of unique elements that reconstruct x, and the corresponding counts for each unique element in x. |
| | | `unique_counts` | Returns the unique elements of an input array x and the corresponding counts for each unique element in x. |
| | | `unique_inverse` | Returns the unique elements of an input array x and the indices from the set of unique elements that reconstruct x. |
| | | `unique_values` | Returns the unique elements of an input array x. |

## Sorting Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `argsort` | Returns the indices that sort an array x along a specified axis. |
| | | `sort` | Returns a sorted copy of an input array x. |

## Statistical Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `cumulative_prod` | Calculates the cumulative product of elements in the input array x. |
| | | `cumulative_sum` | Calculates the cumulative sum of elements in the input array x. |
| Y | [`max`], [`max_axes`] | `max` | Calculates the maximum value of the input array x. |
| Y | [`mean`], [`mean_axes`] | `mean` | Calculates the arithmetic mean of the input array x. |
| Y | [`min`], [`min_axes`] | `min` | Calculates the minimum value of the input array x. |
| Y | [`prod`], [`prod_axes`] | `prod` | Calculates the product of input array x elements. |
| Y | [`std`], [`std_axes`] | `std` | Calculates the standard deviation of the input array x. |
| Y | [`sum`], [`sum_axes`] | `sum` | Calculates the sum of the input array x. |
| Y | [`var`], [`var_axes`] | `var` | Calculates the variance of the input array x. |

## Utility Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`all`], [`all_axes`] | `all` | Tests whether all input array elements evaluate to True along a specified axis. |
| Y | [`any`], [`any_axes`] | `any` | Tests whether any input array element evaluates to True along a specified axis. |
| | | `diff` | Calculates the n-th discrete forward difference along a specified axis. |

## Other Dropped Specifications

We decide to **drop** some supports in Python Array API:
- **Reflected (swapped) operands.** A typical function in python is [`__radd__`](https://docs.python.org/3/reference/datamodel.html#object.__radd__). Reflected operands are not easy to be implemented in rust. I believe that in python, this is realized by checking dynamic object type; this is not friendly to language that requires compilation.
- **Functions related to Python Array API namespace and dlpack.** These routines are mostly for forcing other python packages to be compatible to Python Array API. This is not possible for another language currently.
