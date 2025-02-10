# API document of rstsr-core

`rstsr-core` is the core of RSTSR series:
- It defines data structure and some traits (interface) of tensor, storage, device.
- It realizes two basic devices: [`DeviceCpuSerial`] and [`DeviceFaer`], so `rstsr-core` alone is a functional tensor toolkit library.

If you are more aware of matmul efficiency, or by other considerations (we will try to implement BLAS and Lapack features in future), you may find `DeviceOpenBLAS` in `rstsr-openblas` helpful. We hope to implement more devices in future.

## User document

User document refers to [readthedocs](https://rstsr-book.readthedocs.io/). This document is still in construction.

## API specifications

[API specifications](`api_specification`) is provided for summary of important identifiers (structs, traits and functions) in `rstsr-core`.

Currently, more detailed documentation for each identifiers are still constructing.

We also provide [fullfillment for Array API standard](`array_api_standard`). This fullfillment check shows how much functionalities we have achieved to be a basic tensor (array) library.

For NumPy users, in this meantime, [fullfillment for Array API standard](`array_api_standard`) shows the similar parts of RSTSR and NumPy; and these parts in [API specifications](`api_specification`) shows the difference between RSTSR and NumPy:
- [Tensor Structure and Ownership](`api_specification#tensor-structure-and-ownership`)
- [RSTSR Specific Identifiers](`api_specification#rstsr-specific-identifiers`)
- [Storage-dependent manuplication](`api_specification#storage-dependent-manuplication`)
- [Developer Area](`api_specification#developer-area`)

## Variable naming convention

### Abbreviations of type annotations

| Abbreviation | Meaning |
|--|--|
| `TR` | Tensor type (struct [`TensorBase`]) |
| `T` | Data type (can be `f64`, `Complex<f32>`, etc) |
| `B` | Backend (device) type (applies [`DeviceAPI`]) |
| `D` | Dimensional (applies [`DimAPI`]) |
| `S` | Storage (struct [`storage::Storage`]) |
| `R` | Data with lifetime and mutability (applies [`DataAPI`]) |
| `C` | Content (usually `Vec<T>` for CPU devices) |

### Variable or Names

| Rule | Convention |
|--|--|
| suffix `API` | Traits. <br/> To avoid naming collision with structs or other crates. <br/> Example: [`DeviceAPI`], [`AsArrayAPI`]  |
| suffix `_f` | Fallible functions. <br/> Those functions pairs with a function that panics when error. <br/> Example: [`zeros_f`], [`sin_f`], [`reshape_f`] |
| prefix `to_` or no prefix <br/> (manuplication functions) | Ensures that input is reference. <br/> Example: [`reshape`], [`transpose`], [`to_dim`] |
| prefix `into_` <br/> (some manuplication functions) | Only changes layout, and does not change data with its lifetime and mutability. <br/> Example: [`into_broadcast`], [`into_transpose`], [`into_dyn`] |
| prefix `into_` <br/> output is always [`Tensor`] <br/> (some manuplication functions) | Give owned tensor. <br/> This special case will occur when there is possiblity to generate a new tensor by manuplication function. For these cases, prefix `to_` and `change_` will give output [`TensorCow`], prefix `into_` will give [`Tensor`]. <br/> Example: [`into_shape`], [`into_layout`] |
| prefix `change_` <br/> output is always [`TensorCow`] <br/> (some manuplication functions) | Give copy-on-write tensor by consuming input. <br/> This special case will occur when there is possiblity to generate a new tensor by manuplication function. <br/> Example: [`change_shape`], [`change_layout`] |
| type annotation `Args` | Input tuple type as overloadable parameters. <br/> Example: [`zeros`], [`asarray`] |
| type annotation `Inp` | Additional annotation that rust requires for type deduction. <br/> Example: [`zeros`], [`asarray`] |
