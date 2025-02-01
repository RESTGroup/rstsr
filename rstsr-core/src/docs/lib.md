# API document of rstsr-core

## User document

User document refers to [github pages](https://ajz34.github.io/rstsr-book). Some information of developer guide will also shipped to that document.

## API specifications

This document is the core of this crate.

We provide [fullfillment for Array API standard](`array_api_standard`). This fullfillment check shows how much functionalities we have achieved to be a basic tensor (array) library.

We refer API specifications to [another page](`api_specification`). However, this part is still in construction. In this mean time, we refer [fullfillment for Array API standard](`array_api_standard`) as API specifications.

## Basic structure of rstsr Tensor

Basic structure of this crate:
![rstsr-basic-structure](https://ajz34.github.io/rstsr-book/assets/rstsr-basic-structure.png)

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
| type annotation `Param` | Input tuple type as overloadable parameters. <br/> Example: [`zeros`], [`asarray`] |
| type annotation `Inp` | Additional annotation that rust requires for type deduction. <br/> Example: [`zeros`], [`asarray`] |
