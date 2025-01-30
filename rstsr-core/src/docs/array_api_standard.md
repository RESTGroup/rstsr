# Array API standard Fullfillment

For column status:
- Y: Implemented in rstsr (may not be fully functional like numpy or Python array API specification, but should be enough);
- C: Changed feature (**breaking using experience from numpy**);
- P: Partial implemented in rstsr (not all features in Python array API is implemented);
- D: Features that would be dropped in rstsr.

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
| | | `__floordiv__` | `x1 // x2` |
| **C** | | `__mod__` | `x1 % x2` |
| | | `__pow__` | `x1 ** x2` |
| Y | `+=` | `__iadd__` | `x1 += x2` |
| Y | `-=` | `__isub__` | `x1 -= x2` |
| Y | `*=` | `__imul__` | `x1 *= x2` |
| Y | `/=` | `__itruediv__` | `x1 /= x2` |
| | | `__ifloordiv__` | `x1 //= x2` |
| | | `__ipow__` | `x1 **= x2` |
| Y | `%=` | `__imod__` | `x1 %= x2` |

**Changed feature**
- `__mod__`: We do not use remainder function to represent something like `8 % 3 = 2`, but instead using notation `%` to represent matrix multiplication (`@` in python/numpy).


**Dropped support**
- `__pos__`: In rust, leading `+` is not allowed.

### Array Operators

| status | implementation | Python API | description |
|-|-|-|-|
| **C** | `%` | `__matmul__` | `x1 @ x2` |
| D | | `__imatmul__` | `x1 @= x2` |

**Changed feature**
- `__matmul__`: In rust, there was discussions whether to implement `@` as matrix multiplication (or other operator notations, since `@` has been used in binary operation for pattern matching). Instead we use notation `%` to represent matrix multiplication (`@` in python/numpy). See `__rem__` function for more information.

Dropped support
- `__imatmul__`: Inplace matmul is not convenient to be realized.

### Bitwise Operators

| status | implementation | Python API | description |
|-|-|-|-|
| Y | `!`  | `__invert__` | `~x` |
| Y | `&`  | `__and__` | `x1 & x2` |
| Y | `\|` | `__or__` | `x1 \| x2` |
| Y | `^`  | `__xor__` | `x1 ^ x2` |
| Y | `<<` | `__lshift__` | `x1 << x2` |
| Y | `>>` | `__rshift__` | `x1 >> x2` |
| Y | `&=` | `__iand__` | `x1 &= x2` |
| Y | `\|=`| `__ior__` | `x1 \|= x2` |
| Y | `^=` | `__ixor__` | `x1 ^= x2` |
| Y | `<<=`| `__ilshift__` | `x1 <<= x2` |
| Y | `>>=`| `__irshift__` | `x1 >>= x2` |

### Comparsion Operators

| status | implementation | Python API | description |
|-|-|-|-|
| | | `__lt__` | `x1 < x2` |
| | | `__le__` | `x1 <= x2` |
| | | `__gt__` | `x1 > x2` |
| | | `__ge__` | `x1 >= x2` |
| | | `__eq__` | `x1 == x2` |
| | | `__ne__` | `x1 != x2` |

### Array Object Attributes

| status | implementation | Python API | description |
|-|-|-|-|
| | | `dtype` | Data type of the array elements. |
| | | `device` | Hardware device the array data resides on. |
| | | `mT` | Transpose of a matrix (or a stack of matrices). |
| | | `ndim` | Number of array dimensions (axes). |
| | | `shape` | Array dimensions. |
| | | `size` | Number of elements in an array. |
| | | `T` | Transpose of the array. |

### Methods

| status | implementation | Python API | description |
|-|-|-|-|
| | | `__abs__` | Calculates the absolute value for each element of an array instance. |
| | | `__bool__` | Converts a zero-dimensional array to a Python bool object. |
| | | `__complex__` | Converts a zero-dimensional array to a Python `complex` object. |
| | | `__float__` | Converts a zero-dimensional array to a Python `float` object. |
| | | `__getitem__` | Returns `self[key]`. |
| | | `__index__` | Converts a zero-dimensional integer array to a Python `int` object. |
| | | `__int__` | Converts a zero-dimensional array to a Python `int` object. |
| | | `__setitem__` | Sets `self[key]` to `value`. |
| | | `to_device` | Copy the array from the device on which it currently resides to the specified `device`. |

## Constants

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`core::f64::consts::E`] | [`e`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.constants.e.html) | IEEE 754 floating-point representation of Euler's constant. |
| Y | [`f64::INFINITY`] | [`inf`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.constants.inf.html) | IEEE 754 floating-point representation of (positive) infinity. |
| Y | [`f64::NAN`] | [`nan`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.constants.nan.html) | IEEE 754 floating-point representation of Not a Number (NaN). |
| Y | [`Indexer::Insert`] | [`newaxis`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.constants.newaxis.html) | An alias for None which is useful for indexing arrays. |
| Y | [`core::f64::consts::PI`] | [`pi`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.constants.pi.html) | IEEE 754 floating-point representation of the mathematical constant π. |

## Creation Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`arange`] | [`arange`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html) | Returns evenly spaced values within the half-open interval `[start, stop)` as a one-dimensional array. |
| P | [`asarray`] | [`asarray`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.asarray.html) | Convert the input to an array. |
| Y | [`empty`] | [`empty`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty.html) | Returns an uninitialized array having a specified `shape`. |
| Y | [`empty_like`] | [`empty_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.empty_like.html) | Returns an uninitialized array with the same `shape` as an input array `x`. |
| Y | [`eye`] | [`eye`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.eye.html) | Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere. |
| Y | [`full`] | [`full`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full.html) | Returns a new array having a specified `shape` and filled with `fill_value`. |
| Y | [`full_like`] | [`full_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.full_like.html) | Returns a new array filled with fill_value and having the same `shape` as an input array `x`. |
| Y | [`linspace`] | [`linspace`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.linspace.html) | Returns evenly spaced numbers over a specified interval. |
| | | `meshgrid` | Returns coordinate matrices from coordinate vectors. |
| Y | [`ones`] | [`ones`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones.html) | Returns a new array having a specified shape and filled with ones. |
| Y | [`ones_like`] | [`ones_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.ones_like.html) | Returns a new array filled with ones and having the same `shape` as an input array `x`. |
| | | `tril` | Returns the lower triangular part of a matrix (or a stack of matrices) `x`. |
| | | `triu` | Returns the upper triangular part of a matrix (or a stack of matrices) `x`. |
| Y | [`zeros`] | [`zeros`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros.html) | Returns a new array having a specified `shape` and filled with zeros. |
| Y | [`zeros_like`] | [`zeros_like`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.zeros_like.html) | Returns a new array filled with zeros and having the same `shape` as an input array x. |

**Partial implementation**
- [`asarray`]: This function have different implementations for `Vec<T>`, `[T; N]` and [`Tensor<T, B, D>`]. Different signatures are utilized for different inputs and purposes.

## Data Type

### Data Type Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `astype` | Copies an array to a specified data type irrespective of Type Promotion Rules rules. |
| | | `can_cast` | Determines if one data type can be cast to another data type according Type Promotion Rules rules. |
| | | `finfo` | Machine limits for floating-point data types. |
| | | `iinfo` | Machine limits for integer data types. |
| | | `isdtype` | Returns a boolean indicating whether a provided dtype is of a specified data type "kind". |
| | | `result_type` | Returns the dtype that results from applying the type promotion rules (see Type Promotion Rules) to the arguments. |

### Data Type Categories

| rust trait or struct | data type category | dtypes |
|-|-|-|
| [`num::Num`] | Numeric | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128 |
| | Real-valued | int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64 |
| [`num::Integer`] | Integer | int8, int16, int32, int64, uint8, uint16, uint32, uint64 |
| [`num::complex::ComplexFloat`] | Floating-point | float32, float64, complex64, complex128 |
| [`num::Float`] | Real-valued floating-point | float32, float64 |
| [`num::Complex`] | Complex floating-point | complex64, complex128 |
| [`bool`] | Boolean | bool |


## Element-wise Functions

### Unary Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `abs` | Calculates the absolute value for each element x_i of the input array x. |
| | | `acos` | Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element x_i of the input array x. |
| | | `acosh` | Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element x_i of the input array x. |
| | | `asin` | Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element x_i of the input array x. |
| | | `asinh` | Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element x_i in the input array x. |
| | | `atan` | Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element x_i of the input array x. |
| | | `atanh` | Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element x_i of the input array x. |
| | | `bitwise_invert` | Inverts (flips) each bit for each element x_i of the input array x. |
| | | `ceil` | Rounds each element x_i of the input array x to the smallest (i.e., closest to -infinity) integer-valued number that is not less than x_i. |
| | | `conj` | Returns the complex conjugate for each element x_i of the input array x. |
| | | `cos` | Calculates an implementation-dependent approximation to the cosine for each element x_i of the input array x. |
| | | `cosh` | Calculates an implementation-dependent approximation to the hyperbolic cosine for each element x_i in the input array x. |
| | | `exp` | Calculates an implementation-dependent approximation to the exponential function for each element x_i of the input array x (e raised to the power of x_i, where e is the base of the natural logarithm). |
| | | `expm1` | Calculates an implementation-dependent approximation to exp(x)-1 for each element x_i of the input array x. |
| | | `floor` | Rounds each element x_i of the input array x to the greatest (i.e., closest to +infinity) integer-valued number that is not greater than x_i. |
| | | `imag` | Returns the imaginary component of a complex number for each element x_i of the input array x. |
| | | `isfinite` | Tests each element x_i of the input array x to determine if finite. |
| | | `isinf` | Tests each element x_i of the input array x to determine if equal to positive or negative infinity. |
| | | `isnan` | Tests each element x_i of the input array x to determine whether the element is NaN. |
| | | `log` | Calculates an implementation-dependent approximation to the natural (base e) logarithm for each element x_i of the input array x. |
| | | `log1p` | Calculates an implementation-dependent approximation to log(1+x), where log refers to the natural (base e) logarithm, for each element x_i of the input array x. |
| | | `log2` | Calculates an implementation-dependent approximation to the base 2 logarithm for each element x_i of the input array x. |
| | | `log10` | Calculates an implementation-dependent approximation to the base 10 logarithm for each element x_i of the input array x. |
| | | `logical_not` | Computes the logical NOT for each element x_i of the input array x. |
| | | `negative` | Computes the numerical negative of each element x_i (i.e., y_i = -x_i) of the input array x. |
| | | `positive` | Computes the numerical positive of each element x_i (i.e., y_i = +x_i) of the input array x. |
| | | `real` | Returns the real component of a complex number for each element x_i of the input array x. |
| | | `round` | Rounds each element x_i of the input array x to the nearest integer-valued number. |
| | | `sign` | Returns an indication of the sign of a number for each element x_i of the input array x. |
| | | `signbit` | Determines whether the sign bit is set for each element x_i of the input array x. |
| | | `sin` | Calculates an implementation-dependent approximation to the sine for each element x_i of the input array x. |
| | | `sinh` | Calculates an implementation-dependent approximation to the hyperbolic sine for each element x_i of the input array x. |
| | | `square` | Squares each element x_i of the input array x. |
| | | `sqrt` | Calculates the principal square root for each element x_i of the input array x. |
| | | `tan` | Calculates an implementation-dependent approximation to the tangent for each element x_i of the input array x. |
| | | `tanh` | Calculates an implementation-dependent approximation to the hyperbolic tangent for each element x_i of the input array x. |
| | | `trunc` | Rounds each element x_i of the input array x to the nearest integer-valued number that is closer to zero than x_i. |

### Binary Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `add` | Calculates the sum for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `atan2` | Calculates an implementation-dependent approximation of the inverse tangent of the quotient x1/x2, having domain [-infinity, +infinity] x [-infinity, +infinity] (where the x notation denotes the set of ordered pairs of elements (x1_i, x2_i)) and codomain [-π, +π], for each pair of elements (x1_i, x2_i) of the input arrays x1 and x2, respectively. |
| | | `bitwise_and` | Computes the bitwise AND of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `bitwise_left_shift` | Shifts the bits of each element x1_i of the input array x1 to the left by appending x2_i (i.e., the respective element in the input array x2) zeros to the right of x1_i. |
| | | `bitwise_or` | Computes the bitwise OR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `bitwise_right_shift` | Shifts the bits of each element x1_i of the input array x1 to the right according to the respective element x2_i of the input array x2. |
| | | `bitwise_xor` | Computes the bitwise XOR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `copysign` | Composes a floating-point value with the magnitude of x1_i and the sign of x2_i for each element of the input array x1. |
| | | `divide` | Calculates the division of each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `equal` | Computes the truth value of x1_i == x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `floor_divide` | Rounds the result of dividing each element x1_i of the input array x1 by the respective element x2_i of the input array x2 to the greatest (i.e., closest to +infinity) integer-value number that is not greater than the division result. |
| | | `greater` | Computes the truth value of x1_i > x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `greater_equal` | Computes the truth value of x1_i >= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `hypot` | Computes the square root of the sum of squares for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `less` | Computes the truth value of x1_i < x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `less_equal` | Computes the truth value of x1_i <= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `logaddexp` | Calculates the logarithm of the sum of exponentiations log(exp(x1) + exp(x2)) for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `logical_and` | Computes the logical AND for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `logical_or` | Computes the logical OR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `logical_xor` | Computes the logical XOR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `maximum` | Computes the maximum value for each element x1_i of the input array x1 relative to the respective element x2_i of the input array x2. |
| | | `minimum` | Computes the minimum value for each element x1_i of the input array x1 relative to the respective element x2_i of the input array x2. |
| | | `multiply` | Calculates the product for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `not_equal` | Computes the truth value of x1_i != x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |
| | | `pow` | Calculates an implementation-dependent approximation of exponentiation by raising each element x1_i (the base) of the input array x1 to the power of x2_i (the exponent), where x2_i is the corresponding element of the input array x2. |
| | | `remainder` | Returns the remainder of division for each element x1_i of the input array x1 and the respective element x2_i of the input array x2. |
| | | `subtract` | Calculates the difference for each element x1_i of the input array x1 with the respective element x2_i of the input array x2. |

### Other functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `clip` | Clamps each element x_i of the input array x to the range [min, max]. |

## Indexing Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `take` | Returns elements of an array along an axis. |

## Inspection

| status | implementation | Python API | description |
|-|-|-|-|
| | | `capabilities` | Returns a dictionary of array library capabilities. |
| | | `default_device` | Returns the default device. |
| | | `default_dtypes` | Returns a dictionary containing default data types. |
| | | `devices` | Returns a list of supported devices which are available at runtime. |
| | | `dtypes` | Returns a dictionary of supported Array API data types. |

## Linear Algebra Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `matmul` | Computes the matrix product. |
| | | `matrix_transpose` | Transposes a matrix (or a stack of matrices) x. |
| | | `tensordot` | Returns a tensor contraction of x1 and x2 over specific axes. |
| | | `vecdot` | Computes the (vector) dot product of two arrays. |

## Manipulation Functions

| status | implementation | Python API | description |
|-|-|-|-|
| Y | [`broadcast_arrays`] | [`broadcast_arrays`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_arrays.html) | Broadcasts one or more arrays against one another. |
| Y | [`to_broadcast`] | [`broadcast_to`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.broadcast_to.html) | Broadcasts an array to a specified shape. |
| | | `concat` | Joins a sequence of arrays along an existing axis. |
| Y | [`expand_dims`] | [`expand_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.expand_dims.html) | Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by `axis`. |
| Y | [`flip`] | [`flip`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.flip.html) | Reverses the order of elements in an array along the given axis. |
| | | `moveaxis` | Moves array axes (dimensions) to new positions, while leaving other axes in their original positions. |
| Y | [`transpose`], [`permute_dims`] | [`permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html) | Permutes the axes (dimensions) of an array `x`. |
| | | `repeat` | Repeats each element of an array a specified number of times on a per-element basis. |
| P | [`Tensor::reshape`], [`Tensor::into_shape_assume_contig`] | [`reshape`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.reshape.html) | Reshapes an array without changing its data. |
| | | `roll` | Rolls array elements along a specified axis. |
| P | [`squeeze`] | [`squeeze`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.squeeze.html) | Removes singleton dimensions (axes) from x. |
| | | `stack` | Joins a sequence of arrays along a new axis. |
| | | `tile` | Constructs an array by tiling an input array. |
| | | `unstack` | Splits an array into a sequence of arrays along the given axis. |

**Partial implementation**
- [`squeeze`] accepts one axis as input, instead of accepting multiple axes. This is mostly because output of smaller dimension tensor can be fixed-dimension array ([`DimSmallerOneAPI::SmallerOne`]) when only one axis is passed as argument.
- `reshape`: Currently reshape is work-in-progress. It does not copy array when f/c-contiguous. For numpy, much more cases may not invoke explicit copy when reshape.

## Searching Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `argmax` | Returns the indices of the maximum values along a specified axis. |
| | | `argmin` | Returns the indices of the minimum values along a specified axis. |
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
| | | `cumulative_sum` | Calculates the cumulative sum of elements in the input array x. |
| | | `max` | Calculates the maximum value of the input array x. |
| | | `mean` | Calculates the arithmetic mean of the input array x. |
| | | `min` | Calculates the minimum value of the input array x. |
| | | `prod` | Calculates the product of input array x elements. |
| | | `std` | Calculates the standard deviation of the input array x. |
| | | `sum` | Calculates the sum of the input array x. |
| | | `var` | Calculates the variance of the input array x. |

## Utility Functions

| status | implementation | Python API | description |
|-|-|-|-|
| | | `all` | Tests whether all input array elements evaluate to True along a specified axis. |
| | | `any` | Tests whether any input array element evaluates to True along a specified axis. |

## Other Dropped Specifications

We decide to **drop** some supports in Python Array API:
- **Reflected (swapped) operands.** A typical function in python is [`__radd__`](https://docs.python.org/3/reference/datamodel.html#object.__radd__). Reflected operands are not easy to be implemented in rust. I believe that in python, this is realized by checking dynamic object type; this is not friendly to language that requires compilation.
- **Functions related to Python Array API namespace and dlpack.** These routines are mostly for forcing other python packages to be compatible to Python Array API. This is not possible for another language currently.
