# API Specification of RSTSR

## Notes on API Specification

- "type" column
    - "assoc" is abbreviation for "associated method function", meaning that it can only be called by something like `tensor.foo()` instead of `rt::foo()`.
    - "assoc/fn" means that this function is both defined as associated method and usual function.
    - "core ops" refers to [`core::ops`], indicating that it implements rust language's own operator.
- Function variants
    - Tables in this page will not show function variants. For more details, please see documentation (under construction) of each function / associated methods.
    - Fallible variants:
        - Infallible (panics when error) are not decorated;
        - Fallible (function that returns RSTSR's own [`Result`]) are decorated with `_f` suffix;
        - For example infallible [`asarray`] and fallible [`asarray_f`].
    - Pass-by-value/reference variants:
        - Pass-by-value (consumes the original tensor, `foo(self, ...)`) are decorated with `into_` prefix;
        - Pass-by-reference (does not consumes the original tensor, `foo(&self, ...)`) may have `to_` prefix or none;
        - For example pass-by-value [`into_transpose`] or pass-by-reference [`transpose`].
    - [`TensorCow`] output type variants:
        - Pass-by-value returning [`TensorCow`], decorated with `change_` prefix;
        - Pass-by-value returning [`Tensor`], decorated with `into_` prefix;
        - Pass-by-reference returning [`TensorView`], decorated with `to_` prefix or none;
        - [`reshape`] and [`to_layout`] fits into this category.

## Tensor Structure and Ownership

### Figure illustration of tensor structure

![rstsr-basic-structure](https://rstsr-book.readthedocs.io/latest/assets/images/rstsr-basic-structure-cd99e6e423b65b46c9677fbbbe284760.png)

### Tensor

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [tensorbase][crate::tensorbase] | Defining tensor structs. |
| struct | [`TensorBase<S, D>`] | Basic struct of tensor (storage + layout). |
| alias | [`TensorAny<R, T, B, D>`] | Basic struct of tensor (data with lifetime + device backend + layout). |
| alias | [`Tensor<T, B, D>`] | Tensor that owns its raw data. |
| alias | [`TensorView<'l, T, B, D>`][TensorView] <br/> [`TensorRef<'l, T, B, D>`][TensorRef] | Tensor that shares its raw data by reference. |
| alias | [`TensorMut<'l, T, B, D>`][TensorMut] <br/> [`TensorViewMut<'l, T, B, D>`][TensorViewMut] | Tensor that shares its raw data by mutable reference. |
| alias | [`TensorCow<'l, T, B, D>`][TensorCow] | Tensor either shares its raw data by reference, or owns its raw data (immutable). `Cow` refers to copy-on-write.  |
| alias | [`TensorArc<T, B, D>`][TensorArc] | Tensor with its raw data wrapped by atomic reference-counter pointer ([`Arc`][alloc::sync::Arc]). |

### Tensor Layout

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [layout][crate::layout] | Defining layout of tensor and dimensionality. |
| struct | [`Layout<D>`] | Layout of tensor. |
| trait | [`DimAPI`] | Main basic interface for dimensionality. |
| alias | [`IxD`] | Dynamic dimensionality (alias to `Vec<usize>`). |
| alias | [`Ix<N>`] | Fixed dimensionality (alias to `[usize; N]`). |

### Device

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::device][crate::storage::device] | Defining storage and device. |
| struct | [`DeviceCpuSerial`] | Basic backend that handles computations in single thread. |
| struct | [`DeviceFaer`] | Backend that applies multi-threaded operations (by [rayon](https://github.com/rayon-rs/rayon/)) and efficient matmul (by [faer](https://github.com/sarah-quinones/faer-rs)). |
| struct | [`DeviceCpuRayon`][crate::feature_rayon::DeviceCpuRayon] | Base backend for rayon paralleled devices (device for developer, not user). |
| trait | [`DeviceAPI<T>`] | Main basic interface for device. |

Device is designed to be able extended by other crates. The above devices [`DeviceCpuSerial`] and [`DeviceFaer`] are only special in that they are realized in rstsr-core. We hope that in future, more devices (backends) can be supported.

### Storage

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::device][crate::storage::device] | Defining storage and device. |
| struct | [`Storage<R, T, B>`][crate::storage::device::Storage] | Storage of tensor (data with lifetime + device backend) |

### Tensor Ownership

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::data][crate::storage::data] | Defining data representations (lifetime or the way raw data are stored). |
| struct | [`DataOwned<C>`][crate::storage::data::DataOwned] | Struct wrapper for owned raw data. |
| enum | [`DataRef<'l, C>`][crate::storage::data::DataRef] | Enum wrapper for reference of raw data (or manually-dropped data). |
| enum | [`DataMut<'l, C>`][crate::storage::data::DataMut] | Enum wrapper for mutable reference of raw data (or manually-dropped data). |
| enum | [`DataCow<'l, C>`][crate::storage::data::DataCow] | Enum wrapper for mutable reference of raw data (or manually-dropped data). |
| struct | [`DataArc<C>`][crate::storage::data::DataArc] | Struct wrapper for atomic reference-counted raw data pointer. |
| trait | [`DataAPI`] | Interface of immutable operations for data representations. |
| trait | [`DataCloneAPI`] | Interface of underlying data cloning for data representations. |
| trait | [`DataMutAPI`] | Interface of mutable operations for data representations. |
| trait | [`DataIntoCowAPI<'l>`][DataIntoCowAPI] | Interface for generating [`DataCow<'l, C>`][crate::storage::data::DataCow]. |
| trait | [`DataForceMutAPI`] | Interface for generating mutable reference, ignoring lifetime and borrow checks. |

## Indexing

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc/fn | [`slice`][prim@slice] <br/> [`slice_mut`] | Basic slicing to tensor, generating view of smaller tensor. |
| assoc | [`i`][Tensor::i] <br/> [`i_mut`][Tensor::i_mut] | Alias to [`slice`][prim@slice] and [`slice_mut`]. |
| core ops | operator `[]` <br/> [`Index`] <br/> [`IndexMut`] | Indexing tensor element, giving reference of scalar value (not efficient due to boundary check). |
| assoc | [`index_uncheck`][Tensor::index_uncheck] <br/>[`index_mut_uncheck`][Tensor::index_mut_uncheck] | Indexing tensor element, giving reference of scalar value. |

## RSTSR Specific Identifiers

### Ownership change and transfer

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [ownership_conversion][`crate::tensor::ownership_conversion`] | |
| assoc | [`view`][TensorAny::view] | Get a view of tensor. |
| assoc | [`view_mut`][TensorAny::view_mut] | Get a mutable view of tensor. |
| assoc | [`into_cow`][TensorAny::into_cow] | Convert current tensor into copy-on-write. |
| assoc | [`into_owned`][TensorAny::into_owned] | Convert tensor into owned tensor ([`Tensor`]). Raw data is try to be moved, or the necessary values cloned. |
| assoc | [`into_shared`][TensorAny::into_shared] | Convert tensor into shared tensor ([`TensorArc`]). Raw data is try to be moved, or the necessary values cloned. |
| assoc | [`to_owned`][TensorAny::to_owned] | Clone necessary values to owned tensor ([`Tensor`]) without destroying original tensor. |
| assoc | [`force_mut`][TensorAny::force_mut] | Force generate mutable view of tensor, ignoring lifetime and borrow check. |
| assoc | [`to_vec`][TensorAny::to_vec] | Clone 1-D tensor to `Vec<T>`. |
| assoc | [`into_vec`][TensorAny::into_vec] | Move 1-D tensor to `Vec<T>` if possible, otherwise clone. |
| assoc | [`to_scalar`][TensorAny::to_scalar] | Extract scalar value from tensor that only have one element. |
| assoc | [`as_ptr`][TensorAny::as_ptr] <br/> [`as_mut_ptr`][TensorAny::as_mut_ptr] | Returns pointer to the first element in tensor. |
| fn | [`asarray`] | Convert input (scalar, `Vec<T>`, `&[T]`, tensor) to an array, optionally with shape/layout specified. |
| assoc | [`TensorBase::into_raw_parts`] | Destruct tensor into storage and layout. |
| assoc | [`Storage::into_raw_parts`][crate::storage::device::Storage::into_raw_parts] | Destruct storage into data (with lifetime) and device. |
| assoc | [`DataOwned::into_raw`][crate::storage::data::DataOwned::into_raw] | Destruct owned data and get the raw data (`Vec<T>` for CPU devices). |
| assoc | [`TensorBase::raw`][TensorBase::raw] <br/> [`TensorBase::raw_mut`][TensorBase::raw_mut] | Get reference of raw data. |

<!-- | assoc | [`into_owned_keep_layout`][TensorAny::into_owned_keep_layout] | Convert tensor into owned tensor ([`Tensor`]). Data is either moved or fully cloned. | -->
<!-- | assoc | [`into_shared_keep_layout`][TensorAny::into_shared_keep_layout] | Convert tensor into shared tensor ([`TensorArc`]). Data is either moved or fully cloned. | -->

### Iteration

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [iterator_elem][crate::tensor::iterator_elem] | Tensor iterators that gives elements. |
| assoc | [`iter`][TensorAny::iter] <br/> [`iter_mut`][TensorAny::iter_mut] | Iterate tensor by default ordering (c-prefer or f-prefer). |
| assoc | [`iter_with_order`][TensorAny::iter_with_order] <br/> [`iter_mut_with_order`][TensorAny::iter_mut_with_order] | Iterate tensor with specified order. |
| assoc | [`indexed_iter`][TensorAny::indexed_iter] <br/> [`indexed_iter_mut`][TensorAny::indexed_iter_mut] | Enumerate tensor by default ordering (c-prefer or f-prefer). |
| assoc | [`indexed_iter_with_order`][TensorAny::indexed_iter_with_order] <br/> [`indexed_iter_mut_with_order`][TensorAny::indexed_iter_mut_with_order] | Enumerate tensor with specified order. |
| module | [iterator_axes][crate::tensor::iterator_axes] | Axes iterators that gives smaller tensor views. |
| assoc | [`axes_iter`][TensorAny::axes_iter] <br/> [`axes_iter_mut`][TensorAny::axes_iter_mut] | Iterate tensor by axes by default ordering (c-prefer or f-prefer). |
| assoc | [`axes_iter_with_order`][TensorAny::axes_iter_with_order] <br/> [`axes_iter_mut_with_order`][TensorAny::axes_iter_mut_with_order] | Iterate tensor by axes with specified order. |
| assoc | [`indexed_axes_iter`][TensorAny::indexed_axes_iter] <br/> [`indexed_axes_iter_mut`][TensorAny::indexed_axes_iter_mut] | Enumerate tensor by axes by default ordering (c-prefer or f-prefer). |
| assoc | [`indexed_axes_iter_with_order`][TensorAny::indexed_axes_iter_with_order] <br/> [`indexed_axes_iter_mut_with_order`][TensorAny::indexed_axes_iter_mut_with_order] | Enumerate tensor by axes with specified order. |

### Mapping

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [map_elementwise][crate::tensor::map_elementwise] | Elementwise mapping of tensor. |
| assoc | [`map`][TensorAny::map] <br/> [`map_fnmut`][TensorAny::map_fnmut] | Call function by reference on each element to create a new tensor. |
| assoc | [`mapv`][TensorAny::mapv] <br/> [`mapv_fnmut`][TensorAny::mapv_fnmut] | Call function by value on each element to create a new tensor. |
| assoc | [`mapi`][TensorAny::mapi] <br/> [`mapi_fnmut`][TensorAny::mapi_fnmut] | Modify the tensor in place by calling function by mutable reference on each element. |
| assoc | [`mapvi`][TensorAny::mapvi] <br/> [`mapvi_fnmut`][TensorAny::mapvi_fnmut] | Modify the tensor in place by calling function by reference on each element. |
| assoc | [`mapb`][TensorAny::mapb] <br/> [`mapb_fnmut`][TensorAny::mapb_fnmut] | Map to another tensor and call function by reference on each element to create a new tensor. |
| assoc | [`mapvb`][TensorAny::mapvb] <br/> [`mapvb_fnmut`][TensorAny::mapvb_fnmut] | Map to another tensor and call function by value on each element to create a new tensor. |

### Error handling

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [error][crate::error] | Error handling in RSTSR. |
| enum | [`Error`] | Error type in RSTSR. |
| alias | [`Result<E>`][crate::error::Result] | Result type in RSTSR. |

### Flags

| Type | Identifier | Minimal Description |
|--|--|--|
| flags | [flags][crate::flags] | Flags for tensor. |
| enum | [`FlagOrder`] | The order of tensor. |
| enum | [`FlagDiag`] | Unit-diagonal of matrix. |
| enum | [`FlagSide`] | Side of matrix operation. |
| enum | [`FlagTrans`] | Transposition of matrix operation. |
| enum | [`FlagUpLo`] | Upper/Lower triangular of matrix operation. |
| enum | [`FlagSymm`] | Symmetric of matrix operation. |
| enum | [`TensorIterOrder`] | The policy of tensor iterator. |

## Tensor Manuplication

### Storage-irrelevent manuplication

| Type | Identifier | Minimal Description |
|--|--|--|
| fn | [`broadcast_arrays`] | Broadcasts any number of arrays against each other. |
| assoc/fn | [`to_broadcast`] | Broadcasts an array to a specified shape. |
| assoc/fn | [`expand_dims`] | Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by `axis`. |
| assoc/fn | [`flip`] | Reverses the order of elements in an array along the given axis. |
| assoc/fn | [`permute_dims`] <br/> [`transpose`] | Permutes the axes (dimensions) of an array `x`. |
| assoc/fn | [`reverse_axes`] | Reverse the order of elements in an array along the given axis. |
| assoc/fn | [`swapaxes`] | Interchange two axes of an array. |
| assoc/fn | [`squeeze`] | Removes singleton dimensions (axes) from `x`. |
| assoc/fn | [`to_dim`] <br/> [`to_dyn`] | Convert layout to the other dimension. |
| assoc/fn | [`reshape_assume_contig`] | Assuming contiguous array, reshapes an array without changing its data. |

### Storage-dependent manuplication

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc/fn | [`reshape`] <br/> [`into_shape`] <br/> [`change_shape`] | Reshapes an array without changing its data. |
| assoc/fn | [`to_layout`] <br/> [`into_layout`] <br/> [`change_layout`] | Convert tensor to the other layout. |

### Storage-creation manuplication

These functions are to be realized in future (concat, stack, etc).

## Tensor Creation

| Type | Identifier | Minimal Description |
|--|--|--|
| fn | [`asarray`] | Convert input (scalar, `Vec<T>`, `&[T]`, tensor) to an array, optionally with shape/layout specified. |
| module | [`crate::tensor::creation`] | Creation methods for tensor. |
| fn | [`arange`] | Evenly spaced values within the half-open interval `[start, stop)` as one-dimensional array. |
| fn | [`empty`] | Uninitialized tensor having a specified shape. |
| fn | [`empty_like`] | Uninitialized tensor with the same shape as an input tensor. |
| fn | [`eye`] | Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere. |
| fn | [`full`] | New tensor having a specified shape and filled with given value. |
| fn | [`full_like`] | New tensor filled with given value and having the same shape as an input tensor. |
| fn | [`linspace`] | Evenly spaced numbers over a specified interval. |
| fn | [`ones`] | New tensor filled with ones and having a specified shape. |
| fn | [`ones_like`] | New tensor filled with ones and having the same shape as an input tensor. |
| fn | [`zeros`] | New tensor filled with zeros and having a specified shape. |
| fn | [`zeros_like`] | New tensor filled with zeros and having the same shape as an input tensor. |
| fn | [`tril`] | Returns the lower triangular part of a matrix (or a stack of matrices) x. |
| fn | [`triu`] | Returns the upper triangular part of a matrix (or a stack of matrices) x. |

## Basic Operations

### Unary functions

- Arithmetics: [`neg`], [`not`]

### Binary functions

- Arithmetics: [`add`], [`div`], [`mul`], [`sub`], [`rem`];
- Arithmetics with assignment: [`add_assign`], [`div_assign`], [`mul_assign`], [`rem_assign`], [`sub_assign`];
- Bitwise: [`bitand`], [`bitor`], [`bitxor`], [`shl`], [`shr`];
- Bitwise with assignment: [`bitand_assign`], [`bitor_assign`], [`bitxor_assign`], [`shl_assign`], [`shr_assign`].

<div class="warning">

**NOTE**: [`rem`] can be only called by usual function (`rt::rem` if you have used `rstsr_core::prelude::rt`), but not trait function [`Rem::rem`] or operator `%` (which is overrided for matmul).

Trait function calls like associated methods, so we also do not recommend usage of `tensor.rem(&other)`.

</div>

### Matrix Multiply

Matrix multiply is implemented in many ways. The most useful way is function [`matmul`][`matmul()`] and operator `%`.
- functions [`matmul`][`matmul()`] and [`matmul_with_output`];
- associated methods [`TensorBase::matmul`], [`TensorBase::matmul_from`];
- operator `%`.

<div class="warning">

**NOTE**: Matrix multiplication can also called by trait function [`Rem::rem`], but this is strongly not recommended.

Trait function calls like associated methods, so we also do not recommend usage of `tensor.rem(&other)`.

</div>

## Common Functions

### Unary functions

[`abs`], [`acos`], [`acosh`], [`asin`], [`asinh`], [`atan`], [`atanh`], [`ceil`], [`conj`], [`cos`], [`cosh`], [`exp`], [`expm1`], [`floor`], [`imag`], [`inv`], [`is_finite`], [`is_inf`], [`is_nan`], [`log`], [`log10`], [`log2`], [`real`], [`round`], [`sign`], [`signbit`], [`sin`], [`sinh`], [`sqrt`], [`square`], [`tan`], [`tanh`], [`trunc`]

### Binary functions

[`atan2`], [`copysign`], [`eq`]/[`equal`], [`floor_divide`], [`ge`]/[`greater_equal`], [`gt`]/[`greater`], [`hypot`], [`le`]/[`less_equal`], [`lt`]/[`less`], [`log_add_exp`], [`maximum`], [`minimum`], [`ne`]/[`not_equal`], [`pow`]

### Statistical functions

[`max`], [`mean`], [`min`], [`prod`], [`std`], [`sum`], [`var`]

### Sorting, searching and counting functions

[`argmin`], [`argmax`], [`unraveled_argmin`], [`unraveled_argmax`]

## Developer Area

The above listings of API specifications are mostly for either user usage, or clarafication of most important aspects of the design of RSTSR.

However, there still leaves many public APIs not fully documented or not listed above. Some of them are exposed as developer interfaces.

This part of documentation is under construction.

