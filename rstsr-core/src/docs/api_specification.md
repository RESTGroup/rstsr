# API Specification of crate `rstsr-core`

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
    - With-output variants:
        - Functions decorated with `_with_output` suffix (e.g. [`add_with_output`]) write the result into caller-provided output tensor, instead of allocating a new one.
    - Explicit all-element variants:
        - Reduction families (e.g. [`sum`]) are also decorated with `_all` suffix for explicitly reducing all elements into a scalar, which behaves the same as the undecorated function (e.g. [`sum_all`]).
    - Argument groups:
        - A set of related arguments travels as one tuple parameter, e.g. `rt::repeat(x, (repeats, axis))`, `rt::sort(x, (axis, descending, stable))`; `rt::func(x, y)` and `rt::func((x, y))` are different signatures (two parameters vs one tuple parameter).
        - The group types ([`RepeatArgs`], [`RollArgs`], [`SortArgs`], [`SearchSortedArgs`], ...) carry the overloads (`(a, b)`, `a`, `()`, `None`, ...), so one signature covers every documented call shape. Where `()` is accepted, `None` is accepted as well and means the same.
    - Reduction argument variants:
        - Reduction families accept grouped arguments (`axes`, `keepdims`, and for var/std `correction`) by the `_with_args` suffix (e.g. [`sum_with_args`]), with the argument types [`ReduceArgs`] / [`VarArgs`];
        - Accumulating families (sum, prod, mean, var, std) also accept an explicit output dtype by the `_with_dtype` suffix (e.g. [`sum_with_dtype`]); the fold accumulates in the requested dtype.

## Tensor Structure and Ownership

### Figure illustration of tensor structure

![rstsr-basic-structure](https://rstsr-book.readthedocs.io/latest/assets/images/rstsr-basic-structure-cd99e6e423b65b46c9677fbbbe284760.png)

### Tensor

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [tensorbase][rstsr_core::tensorbase] | Defining tensor structs. |
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
| module | [layout][rstsr_common::layout] | Defining layout of tensor and dimensionality. |
| struct | [`Layout<D>`] | Layout of tensor. |
| trait | [`DimAPI`] | Main basic interface for dimensionality. |
| alias | [`IxD`] | Dynamic dimensionality (alias to `Vec<usize>`). |
| alias | [`Ix<N>`] | Fixed dimensionality (alias to `[usize; N]`). |

### Device

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::device][rstsr_core::storage::device] | Defining storage and device. |
| struct | [`DeviceCpuSerial`] | Basic backend that handles computations in single thread. |
| struct | [`DeviceFaer`] | Backend that applies multi-threaded operations (by [rayon](https://github.com/rayon-rs/rayon/)) and efficient matmul (by [faer](https://github.com/sarah-quinones/faer-rs)). |
| struct | [`DeviceCpuRayon`][rstsr_core::feature_rayon::DeviceCpuRayon] | Base backend for rayon paralleled devices (device for developer, not user). |
| alias | [`DeviceCpu`] | Default device alias: [`DeviceFaer`] if cargo feature `faer_as_default` is enabled, otherwise [`DeviceCpuSerial`]. |
| trait | [`DeviceAPI<T>`] | Main basic interface for device. |

Device is designed to be able extended by other crates. The above devices [`DeviceCpuSerial`] and [`DeviceFaer`] are only special in that they are realized in rstsr-core. We hope that in future, more devices (backends) can be supported.

### Storage

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::device][rstsr_core::storage::device] | Defining storage and device. |
| struct | [`Storage<R, T, B>`][rstsr_core::storage::device::Storage] | Storage of tensor (data with lifetime + device backend) |

### Tensor Ownership

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [storage::data][rstsr_core::storage::data] | Defining data representations (lifetime or the way raw data are stored). |
| struct | [`DataOwned<C>`][rstsr_core::storage::data::DataOwned] | Struct wrapper for owned raw data. |
| enum | [`DataRef<'l, C>`][rstsr_core::storage::data::DataRef] | Enum wrapper for reference of raw data (or manually-dropped data). |
| enum | [`DataMut<'l, C>`][rstsr_core::storage::data::DataMut] | Enum wrapper for mutable reference of raw data (or manually-dropped data). |
| enum | [`DataCow<'l, C>`][rstsr_core::storage::data::DataCow] | Enum wrapper for mutable reference of raw data (or manually-dropped data). |
| struct | [`DataArc<C>`][rstsr_core::storage::data::DataArc] | Struct wrapper for atomic reference-counted raw data pointer. |
| trait | [`DataAPI`] | Interface of immutable operations for data representations. |
| trait | [`DataCloneAPI`] | Interface of underlying data cloning for data representations. |
| trait | [`DataMutAPI`] | Interface of mutable operations for data representations. |
| trait | [`DataIntoCowAPI<'l>`][DataIntoCowAPI] | Interface for generating [`DataCow<'l, C>`][rstsr_core::storage::data::DataCow]. |
| trait | [`DataForceMutAPI`] | Interface for generating mutable reference, ignoring lifetime and borrow checks. |

## Indexing

### Basic Indexing

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc | [`i`][Tensor::i] <br/> [`i_mut`][Tensor::i_mut] | Alias to [`slice`](slice()) and [`slice_mut`]. Recommended over using [`slice`](slice()) in most cases. |
| assoc/fn | [`slice`](slice()) <br/> [`slice_mut`] <br/> [`into_slice`] | Basic slicing to tensor, generating view of smaller tensor. |
| macro | [`slice!`] | Construct basic-indexing slice by (`start`, `stop`, `step`), e.g. `slice!(1, -4, 2)`. |
| core ops | operator `[]` <br/> [`Index`] <br/> [`IndexMut`] | Indexing tensor element, giving reference of scalar value (not efficient due to boundary check). |
| assoc | [`index_uncheck`][Tensor::index_uncheck] <br/>[`index_mut_uncheck`][Tensor::index_mut_uncheck] | Indexing tensor element, giving reference of scalar value. |
| assoc/fn | [`diagonal`] <br/> [`diagonal_mut`] <br/> [`into_diagonal`] <br/> [`into_diagonal_mut`] | View (or mutable view) of the specified diagonal of a matrix (or a stack of matrices). |

### Advanced Indexing

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc/fn | [`bool_select`] | Returns a new tensor, which indexes the input tensor along dimension `axis` using the boolean entries in `mask`. |
| assoc/fn | [`index_select`] | Returns a new tensor, which indexes the input tensor along dimension `axis` using the entries in `indices`. |
| assoc/fn | [`array_index`] <br/> [`array_index_assign`] | Array indexing (fancy indexing) by integer arrays; the in-place scatter counterpart. |
| assoc/fn | [`mask_select`] <br/> [`mask_assign`] <br/> [`mask_fill`] | Select / assign / fill the elements where a boolean `mask` is true. |
| assoc/fn | [`take`] | Take elements from an array along an axis. |
| assoc/fn | [`take_along_axis`] <br/> [`put_along_axis`] | Gather / in-place scatter along `axis` by an index tensor. |
| assoc/fn | [`index_put`] | In-place scatter of values along one axis by an index list. |

## RSTSR Specific Identifiers

### Ownership change and transfer

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [ownership_conversion][`rstsr_core::tensor::ownership_conversion`] | |
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
| assoc/fn | [`astype`] <br/> [`astype_f`] | Cast a tensor to another data type; a view when the dtype is unchanged, otherwise a dense copy in the input's memory order. |
| assoc | [`TensorBase::into_raw_parts`] | Destruct tensor into storage and layout. |
| assoc | [`Storage::into_raw_parts`][rstsr_core::storage::device::Storage::into_raw_parts] | Destruct storage into data (with lifetime) and device. |
| assoc | [`DataOwned::into_raw`][rstsr_core::storage::data::DataOwned::into_raw] | Destruct owned data and get the raw data (`Vec<T>` for CPU devices). |
| assoc | [`TensorBase::raw`][TensorBase::raw] <br/> [`TensorBase::raw_mut`][TensorBase::raw_mut] | Get reference of raw data. |

<!-- | assoc | [`into_owned_keep_layout`][TensorAny::into_owned_keep_layout] | Convert tensor into owned tensor ([`Tensor`]). Data is either moved or fully cloned. | -->
<!-- | assoc | [`into_shared_keep_layout`][TensorAny::into_shared_keep_layout] | Convert tensor into shared tensor ([`TensorArc`]). Data is either moved or fully cloned. | -->

### Device transfer

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [device_conversion][rstsr_core::tensor::device_conversion] | Device transfer of tensor. |
| trait | [`TensorDeviceChangeAPI`] | Change the device of tensor: [`TensorDeviceChangeAPI::to_device`] (borrowing form, cloning to target device), [`TensorDeviceChangeAPI::into_device`] (consuming form, returning owned tensor on target device), [`TensorDeviceChangeAPI::change_device`] (consuming form, keeping storage representation). |

### Iteration

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [iterator_elem][rstsr_core::tensor::iterator_elem] | Tensor iterators that gives elements. |
| assoc | [`iter`][TensorView::iter] <br/> [`iter_mut`][TensorMut::iter_mut] | Iterate tensor by default ordering (c-prefer or f-prefer). |
| assoc | [`iter_with_order`][TensorView::iter_with_order] <br/> [`iter_mut_with_order`][TensorMut::iter_mut_with_order] | Iterate tensor with specified order. |
| assoc | [`indexed_iter`][TensorView::indexed_iter] <br/> [`indexed_iter_mut`][TensorMut::indexed_iter_mut] | Enumerate tensor by default ordering (c-prefer or f-prefer). |
| assoc | [`indexed_iter_with_order`][TensorView::indexed_iter_with_order] <br/> [`indexed_iter_mut_with_order`][TensorMut::indexed_iter_mut_with_order] | Enumerate tensor with specified order. |
| module | [iterator_axes][rstsr_core::tensor::iterator_axes] | Axes iterators that gives smaller tensor views. |
| assoc | [`axes_iter`][TensorView::axes_iter] <br/> [`axes_iter_mut`][TensorMut::axes_iter_mut] | Iterate tensor by axes by default ordering (c-prefer or f-prefer). |
| assoc | [`axes_iter_with_order`][TensorView::axes_iter_with_order] <br/> [`axes_iter_mut_with_order`][TensorMut::axes_iter_mut_with_order] | Iterate tensor by axes with specified order. |
| assoc | [`indexed_axes_iter`][TensorView::indexed_axes_iter] <br/> [`indexed_axes_iter_mut`][TensorMut::indexed_axes_iter_mut] | Enumerate tensor by axes by default ordering (c-prefer or f-prefer). |
| assoc | [`indexed_axes_iter_with_order`][TensorView::indexed_axes_iter_with_order] <br/> [`indexed_axes_iter_mut_with_order`][TensorMut::indexed_axes_iter_mut_with_order] | Enumerate tensor by axes with specified order. |

### Mapping

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [map_elementwise][rstsr_core::tensor::map_elementwise] | Elementwise mapping of tensor. |
| assoc | [`map`][TensorAny::map] <br/> [`map_fnmut`][TensorAny::map_fnmut] | Call function by reference on each element to create a new tensor. |
| assoc | [`mapv`][TensorAny::mapv] <br/> [`mapv_fnmut`][TensorAny::mapv_fnmut] | Call function by value on each element to create a new tensor. |
| assoc | [`mapi`][TensorAny::mapi] <br/> [`mapi_fnmut`][TensorAny::mapi_fnmut] | Modify the tensor in place by calling function by mutable reference on each element. |
| assoc | [`mapvi`][TensorAny::mapvi] <br/> [`mapvi_fnmut`][TensorAny::mapvi_fnmut] | Modify the tensor in place by calling function by reference on each element. |
| assoc | [`mapb`][TensorAny::mapb] <br/> [`mapb_fnmut`][TensorAny::mapb_fnmut] | Map to another tensor and call function by reference on each element to create a new tensor. |
| assoc | [`mapvb`][TensorAny::mapvb] <br/> [`mapvb_fnmut`][TensorAny::mapvb_fnmut] | Map to another tensor and call function by value on each element to create a new tensor. |

### Error handling

| Type | Identifier | Minimal Description |
|--|--|--|
| module | [error][rstsr_common::error] | Error handling in RSTSR. |
| enum | [`Error`] | Error type in RSTSR. |
| alias | [`Result<E>`][rstsr_common::error::Result] | Result type in RSTSR. |

### Flags

| Type | Identifier | Minimal Description |
|--|--|--|
| flags | [flags][rstsr_common::flags] | Flags for tensor. |
| enum | [`FlagOrder`] | The order of tensor ([`RowMajor`], [`ColMajor`]). |
| enum | [`FlagDiag`] | Unit-diagonal of matrix. |
| enum | [`FlagSide`] | Side of matrix operation. |
| enum | [`FlagTrans`] | Transposition of matrix operation. |
| enum | [`FlagUpLo`] | Upper/Lower triangular of matrix operation. |
| enum | [`FlagSymm`] | Symmetric of matrix operation. |
| alias | [`TensorOrder`] | Alias to [`FlagOrder`]. |
| enum | [`TensorIterOrder`] | The policy of tensor iterator. |

## Tensor Manipulation

### Storage-irrelevant manipulation

| Type | Identifier | Minimal Description |
|--|--|--|
| fn | [`broadcast_arrays`] | Broadcasts any number of arrays against each other. |
| fn | [`broadcast_shapes`] | Broadcasts shapes against each other and returns the resulting shape. |
| assoc/fn | [`to_broadcast`] <br/> [`broadcast_to`][TensorAny::broadcast_to] <br/> [`into_broadcast`] | Broadcasts an array to a specified shape. |
| assoc/fn | [`expand_dims`] <br/> [`unsqueeze`][TensorAny::unsqueeze] | Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by `axis`. |
| assoc/fn | [`atleast_1d`] <br/> [`atleast_2d`] <br/> [`atleast_3d`] <br/> [`into_atleast_1d`] <br/> [`into_atleast_2d`] <br/> [`into_atleast_3d`] | View the input as a tensor with at least 1 / 2 / 3 dimensions (the `into_*` forms consume ownership). |
| assoc/fn | [`flip`] | Reverses the order of elements in an array along the given axis. |
| assoc/fn | [`moveaxis`] | Moves array axes (dimensions) to new positions, while leaving other axes in their original positions. |
| assoc/fn | [`searchsorted`] | Finds insertion positions of values into a sorted 1-D array (`side`, `sorter` arguments in [`SearchSortedArgs`]). |
| assoc/fn | [`unique_values`] <br/> [`unique_counts`] <br/> [`unique_inverse`] <br/> [`unique_all`] | Unique values of the flattened tensor, with counts / inverse mapping / all fields (named structs `UniqueCounts`, `UniqueInverse`, `UniqueAll`). |
| assoc/fn | [`isin`] | Element-wise membership of `x1` in `x2` (`invert` argument). |
| assoc/fn | [`nonzero`] | Indices of nonzero elements, one 1-D coordinate tensor per dimension. |
| assoc/fn | [`take_along_axis`] | Gather along an axis with an index tensor (companion of `argsort`). |
| assoc/fn | [`repeat`] | Repeat elements of a tensor along an axis (or the flattened row-major sequence). |
| assoc/fn | [`roll`] | Roll array elements along a given axis (the flattened form visits row-major). |
| assoc/fn | [`tile`] | Construct an array by tiling an input array. |
| assoc/fn | [`permute_dims`] <br/> [`transpose`] <br/> [`matrix_transpose`] <br/> [`into_matrix_transpose`] | Permutes the axes (dimensions) of an array `x`. |
| assoc/fn | [`reshape_with_args`] | Reshapes the given tensor to the specified shape, with argument specifying the order and whether to copy data. |
| assoc/fn | [`reverse_axes`] | Reverses the order of the axes (dimensions) of an array. |
| assoc/fn | [`swapaxes`] | Interchange two axes of an array. |
| assoc/fn | [`squeeze`] | Removes singleton dimensions (axes) from `x`. |
| assoc/fn | [`to_compatible_shape`] <br/> [`into_compatible_shape`] | Reshape to the specified shape if the layout is compatible. |
| assoc/fn | [`to_dim`] <br/> [`to_dyn`] | Convert layout to the other dimension. |
| assoc | [`reshapeable_without_copy`] | Whether the tensor can be reshaped to a given shape without copying data. |
| assoc | `into_*` | Ownership-consuming layout-only variants of the operations above: [`into_flip`], [`into_swapaxes`], [`into_moveaxis`], [`into_permute_dims`], [`into_reverse_axes`], [`into_expand_dims`], [`into_squeeze`], [`into_unsqueeze`], [`into_transpose`], [`into_dim`], [`into_dyn`]. |

### Storage-dependent manipulation

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc/fn | [`reshape`] <br/> [`to_shape`] <br/> [`into_shape`] <br/> [`change_shape`] | Reshapes an array without changing its data. |
| assoc/fn | [`to_layout`] <br/> [`into_layout`] <br/> [`change_layout`] | Convert tensor to the other layout. |
| assoc/fn | [`to_contig`] <br/> [`into_contig`] <br/> [`change_contig`] | Convert tensor to contiguous layout (C or F order). |
| assoc/fn | [`to_prefer`] <br/> [`into_prefer`] <br/> [`change_prefer`] | Convert tensor to preferred layout only if not already contiguous. |

### Storage-creation manipulation

| Type | Identifier | Minimal Description |
|--|--|--|
| fn | [`meshgrid`] | Returns coordinate matrices from coordinate vectors. |
| assoc/fn | [`diag`] | Returns specified diagonal of a matrix (or a stack of matrices), or constructs a diagonal matrix from a 1-D tensor. |
| fn | [`concat`](concat()) <br/> [`concatenate`](concatenate()) | Join a sequence of arrays along an existing axis. |
| fn | [`stack`] | Joins a sequence of arrays along a new axis. |
| fn | [`hstack`] | Stack tensors in sequence horizontally (column-wise). |
| fn | [`vstack`] | Stack tensors in sequence vertically (row-wise). |
| assoc/fn | [`unstack`] | Splits an array into a sequence of arrays along the given axis. |

## Tensor Creation

| Type | Identifier | Minimal Description |
|--|--|--|
| fn | [`asarray`] | Convert input (scalar, `Vec<T>`, `&[T]`, tensor) to an array, optionally with shape/layout specified. |
| module | [`rstsr_core::tensor::creation`] | Creation methods for tensor. |
| fn | [`arange`] | Evenly spaced values within the half-open interval `[start, stop)` as one-dimensional array. |
| fn | [`assume_init`] | Convert tensor of [`core::mem::MaybeUninit`] elements into initialized tensor (requires `unsafe` at call site). |
| fn | [`empty`] | Uninitialized tensor having a specified shape (requires `unsafe` at call site). |
| fn | [`empty_like`] | Uninitialized tensor with the same shape as an input tensor (requires `unsafe` at call site). |
| fn | [`eye`] | Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere. |
| fn | [`from_scalar`] | Returns a 0-dimensional (scalar) tensor of a given value. |
| fn | [`full`] | New tensor having a specified shape and filled with given value. |
| fn | [`full_like`] | New tensor filled with given value and having the same shape as an input tensor. |
| fn | [`linspace`] | Evenly spaced numbers over a specified interval. |
| fn | [`uninit`] | Uninitialized tensor of [`core::mem::MaybeUninit`] elements; after initialization, convert into plain tensor by [`assume_init`]. |
| fn | [`ones`] | New tensor filled with ones and having a specified shape. |
| fn | [`ones_like`] | New tensor filled with ones and having the same shape as an input tensor. |
| fn | [`zeros`] | New tensor filled with zeros and having a specified shape. |
| fn | [`zeros_like`] | New tensor filled with zeros and having the same shape as an input tensor. |
| fn | [`tril`] | Returns the lower triangular part of a matrix (or a stack of matrices) x. |
| fn | [`triu`] | Returns the upper triangular part of a matrix (or a stack of matrices) x. |
| macro | [`tensor_from_nested`] | Create a tensor from a nested array literal (for development purposes). |

## Basic Operations

### Unary functions

- Arithmetics: [`neg`], [`ext_neg`], [`positive`], [`not`]

[`ext_neg`]: unary negation with NumPy semantics — preserves the input dtype and wraps unsigned integers around the two's-complement modulus (the Array API `negative`; the `-` operator and [`neg`] cover signed integers, floats and complex, but not unsigned dtypes).

### Binary functions

- Arithmetics: [`add`], [`div`], [`mul`], [`sub`], [`rem`];
- Arithmetics with dtype promotion: [`ext_add`], [`ext_sub`], [`ext_mul`], [`ext_div`];
- Arithmetics with assignment: [`add_assign`], [`div_assign`], [`mul_assign`], [`rem_assign`], [`sub_assign`];
- Bitwise: [`bitand`], [`bitor`], [`bitxor`], [`shl`], [`shr`];
- Bitwise with dtype promotion: [`ext_bitand`], [`ext_bitor`], [`ext_bitxor`], [`ext_shl`], [`ext_shr`];
- Bitwise with assignment: [`bitand_assign`], [`bitor_assign`], [`bitxor_assign`], [`shl_assign`], [`shr_assign`].

The `ext_` binary functions promote mixed-dtype operands to their common data type (following [`DTypePromoteAPI`], the same rule as NumPy) and compute the result in that type: they back the Array API `add` / `subtract` / `multiply` / `divide` and the `bitwise_*` functions. The same-dtype functions ([`add`], [`bitand`], ...) and the Rust operators (`+`, `-`, `*`, `/`, `&`, `|`, `^`, `<<`, `>>`) instead require matching operand dtypes. [`ext_shl`] / [`ext_shr`] additionally follow the Array API shift rule at the bit-width boundary (a left shift by at least the bit width yields `0`; a right shift saturates to the sign), unlike Rust's masking `<<` / `>>`.

<div class="warning">

**Prefer the Rust operators** (`+`, `-`, `*`, `/`, `&`, `|`, `^`, `<<`, `>>`) and the same-dtype functions ([`add`], [`mul`], [`bitand`], ...) in Rust code. The `ext_` binary functions exist primarily to fulfill the Array API's mixed-dtype promotion contract; they are **not** the idiomatic rstsr surface, and the operators/functions above are the recommended way when the operands already share a dtype.

</div>

<div class="warning">

**NOTE**: [`rem`] can be only called by usual function (`rt::rem` if you have used `rstsr_core::prelude::rt`), but not trait function [`Rem::rem`] or operator `%` (which is overrided for matmul).

Trait function calls like associated methods, so we also do not recommend usage of `tensor.rem(&other)`.

</div>

### Matrix Multiply

Matrix multiply is implemented in many ways. The most useful way is function [`matmul`][`matmul()`] and operator `%`.
- functions [`matmul`][`matmul()`], [`matmul_from`] and [`matmul_with_output`];
- function [`ext_matmul`] for mixed-dtype operands (dtype promotion);
- associated methods [`TensorBase::matmul`], [`TensorBase::matmul_from`] and [`TensorBase::ext_matmul`];
- operator `%`.

**The [`ext_matmul`] entry exists only for array-API compliance**, not as the idiomatic rstsr surface: it accepts mixed-dtype operands and computes in their promoted common data type (the same rule as NumPy). Prefer [`matmul`] (or the operator `%`) whenever the operands already share a dtype.

<div class="warning">

**NOTE**: Matrix multiplication can also called by trait function [`Rem::rem`], but this is strongly not recommended.

Trait function calls like associated methods, so we also do not recommend usage of `tensor.rem(&other)`.

</div>

### Tensor Contraction

| Type | Identifier | Minimal Description |
|--|--|--|
| assoc/fn | [`vecdot`], [`ext_vecdot`] <br/> [`vecdot_from`] | Computes the (vector) dot product of two arrays. |
| assoc/fn | [`tensordot`], [`ext_tensordot`] <br/> [`tensordot_from`] | Contracts two arrays over specified axes. |
| assoc/fn | [`outer`], [`ext_outer`] <br/> [`outer_from`] | Computes the outer product of two one-dimensional arrays. |
| assoc/fn | [`trace`] | Sums along the diagonal of a tensor. |
| assoc/fn | [`cross`], [`ext_cross`] | Computes the cross product of 3-element vectors along one axis. |

**The `ext_` forms ([`ext_vecdot`], [`ext_tensordot`], [`ext_outer`], [`ext_cross`]) exist only for array-API compliance**, not as the idiomatic rstsr surface: they accept mixed-dtype operands and compute in their promoted common data type (the same rule as NumPy), unlike the same-dtype [`vecdot`] / [`tensordot`] / [`outer`] / [`cross`], which require matching operand dtypes. Prefer the same-dtype entries whenever the operands already share a dtype.

<div class="warning">

**`rt::tensordot` and `rt::ext_tensordot` are not recommended** when efficiency matters: general axis pairs run on a naive kernel with no BLAS-backed path. Prefer [`rt::tblis::einsum`](https://docs.rs/rstsr-tblis/latest/rstsr_tblis/einsum_impl/fn.einsum.html), enabled in the main crate `rstsr` with feature `tblis` — this requires the user to build and install the TBLIS library themselves.

`einsum` itself is not implemented in `rstsr-core`.

</div>

## Common Functions

### Unary functions

[`abs`], [`acos`], [`acosh`], [`asin`], [`asinh`], [`atan`], [`atanh`], [`ceil`], [`conj`], [`cos`], [`cosh`], [`exp`], [`expm1`], [`floor`], [`imag`], [`inv`], [`is_finite`], [`is_inf`], [`is_nan`], [`log`], [`log10`], [`log1p`], [`log2`], [`real`], [`reciprocal`], [`round`], [`sign`], [`signbit`], [`sin`], [`sinh`], [`sqrt`], [`square`], [`tan`], [`tanh`], [`trunc`]

### Binary functions

[`atan2`], [`copysign`], [`eq`]/[`equal`], [`floor_divide`], [`ge`]/[`greater_equal`], [`gt`]/[`greater`], [`hypot`], [`le`]/[`less_equal`], [`lt`]/[`less`], [`log_add_exp`], [`maximum`], [`minimum`], [`ne`]/[`not_equal`], [`nextafter`], [`pow`]

Aliases: [`equal_than`], [`greater_than`], [`less_than`], [`greater_equal_to`], [`less_equal_to`], and [`not_equal_to`] (for [`equal`], [`greater`], [`less`], [`greater_equal`], [`less_equal`], and [`not_equal`], respectively).

### Statistical functions

[`max`]/[`max_axes`], [`mean`]/[`mean_axes`], [`min`]/[`min_axes`], [`prod`]/[`prod_axes`], [`std`](std())/[`std_axes`], [`sum`]/[`sum_axes`], [`var`]/[`var_axes`], [`l2_norm`]/[`l2_norm_axes`]

Args/dtype variants: all of the above (plus [`all`], [`any`], argmin/argmax families, [`count_nonzero`], [`unraveled_argmin`]/[`unraveled_argmax`]) also have `_with_args` forms taking [`ReduceArgs`] (axes + keepdims); var/std take [`VarArgs`] (adds array-api `correction`); sum/prod/mean/var/std additionally have `_with_dtype` forms with an explicit output dtype.

Closeness testing: [`allclose`](allclose()) (explicit all-element form [`allclose_all`]), or macro [`allclose!`].


Cumulative functions (scans, not reductions; output keeps the input shape): [`cumulative_sum`], [`cumulative_prod`], with [`CumulativeArgs`] (`axis` + `include_initial`) and `_with_dtype` forms.


Custom user reduction (expert-level): [`reduce_all`], [`reduce_axes`], [`reduce_with_args`] with user-provided init/fold/combine/finalize closures. 

### Sorting, searching and counting functions

[`argmin`]/[`argmin_axes`], [`argmax`]/[`argmax_axes`], [`count_nonzero`]/[`count_nonzero_axes`], [`unraveled_argmin`]/[`unraveled_argmin_axes`], [`unraveled_argmax`]/[`unraveled_argmax_axes`], each with `_with_args` forms; NaN-skipping [`nanargmin`]/[`nanargmin_axes`], [`nanargmax`]/[`nanargmax_axes`] (NumPy semantics). Select function [`where`] (boolean condition, NumPy three-argument form); and clip function [`clip`] (`(lo, hi)` bounds, each optional)

Sorting family: [`sort`], [`argsort`] with [`SortArgs`] (`axis` + `descending` + `stable`), plus expert comparator variants [`sort_custom`]/[`argsort_custom`]. Searching: [`searchsorted`] with [`SearchSortedArgs`]; [`nonzero`]. Gathering: [`take_along_axis`]. Discrete differences: [`diff`] (`axis` + `n` + optional `prepend`/`append`).

### Utility functions

[`all`]/[`all_axes`], [`any`]/[`any_axes`]

## Utility Types and Argument Groups

Types that appear in the signatures above: the axis arguments, the indexer types
behind indexing, and the argument-group structs that let one function signature
cover several call shapes.

| Type | Identifier | Minimal Description |
|--|--|--|
| enum | [`AxisIndex`] | Single-axis argument ([`AxisIndex::None`] means the last axis). |
| enum | [`AxesIndex`] | Multi-axis (or no-axis) argument. |
| enum | [`SliceI`] | One basic-indexing slice entry (`start`, `stop`, `step`). |
| enum | [`Indexer`] | One per-axis basic indexer (integer, slice, new-axis, ellipsis). |
| enum | [`ArrayIndexer`] <br/> struct [`ArrayIndexArgs`] | Per-axis array-indexing indexer and its argument group. |
| struct | [`ClipArgs`] <br/> enum [`ClipSide`] / [`ClipSideArg`] | Clip bounds and bounded side(s). |
| enum | [`SearchSide`] | `searchsorted` side (`left` / `right`). |
| struct | [`ReshapeArgs`], [`RepeatArg`] / [`RepeatArgs`], [`RollArgs`], [`SortArgs`], [`SearchSortedArgs`], [`ReduceArgs`], [`VarArgs`], [`CumulativeArgs`] | Argument groups carried as one tuple parameter. |
| struct | [`UniqueCounts`], [`UniqueInverse`], [`UniqueAll`] | Named results of the `unique_*` family. |

## Developer Area

The above listings of API specifications are mostly for either user usage, or clarification of most important aspects of the design of RSTSR.

However, there still leaves many public APIs not fully documented or not listed above. Some of them are exposed as developer interfaces.

This part of documentation is under construction.

