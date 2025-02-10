# RSTSR: An n-Dimension Rust Tensor Toolkit

<center>

| Resources | Badges |
|--|--|
| User Document | [![User Documentation](https://readthedocs.org/projects/rstsr-book/badge/?version=latest)](https://rstsr-book.readthedocs.io/latest/) |
| API Document | [![API Documentation](https://docs.rs/rstsr/badge.svg)](https://docs.rs/rstsr) |
| Crate | [![Crate](https://img.shields.io/crates/v/rstsr.svg)](https://crates.io/crates/rstsr) |

</center>

Welcome to RSTSR, a n-dimensional tensor toolkit library, in native rust.

This crate will be a building block for scientific computation in native Rust, similar to NumPy of Python.

## Features

- Simple syntex (looks like NumPy, and some core concepts from rust crate [ndarray](https://github.com/rust-ndarray/ndarray/)).
- % (remainder) as matrix multiplication (you can `&a % &b` to perform `a.matmul(&b)`).
- Allow different devices in framework. Currently supports `DeviceFaer` and `DeviceOpenBLAS`.
    - We will try to support CUDA and HIP in near future.
- Full support of n-dimensional, broadcasting, basic slicing, reshape.
- Fast on multi-threading CPU.
    - Matmul is provided by backends (such as [faer](https://github.com/sarah-quinones/faer-rs/) or [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/)).
    - Other cases (summation, element-wise operations) are on-par or even much faster than NumPy (by fast layout iterators and [rayon](https://github.com/rayon-rs/rayon/) threading).

## Illustrative Example

To start with, you may try to run the following code:

```rust
use rstsr::prelude::*;

// 3x2 matrix with c-contiguous memory layout
let a = rt::asarray((vec![6., 2., 7., 4., 8., 5.], [3, 2].c()));

// 2x4x3 matrix by arange and reshaping
let b = rt::arange(24.);
let b = b.reshape((-1, 4, 3));
// in one line, you can also
// let b = rt::arange(24.).into_shape((-1, 4, 3));

// broadcasted matrix multiplication
let c = &b % &a;

// print the result
println!("{:6.1}", c)
// output:
// [[[   23.0   14.0]
//   [   86.0   47.0]
//   [  149.0   80.0]
//   [  212.0  113.0]]
//
//  [[  275.0  146.0]
//   [  338.0  179.0]
//   [  401.0  212.0]
//   [  464.0  245.0]]]

// print layout of the result
println!("{:?}", c.layout());
// output:
// 3-Dim (dyn), contiguous: Cc
// shape: [2, 4, 2], stride: [8, 2, 1], offset: 0
```

## Short FAQs

> **Why RSTSR? There seems many numeric and machine-learning libraries in rust already.**

We need a numeric library that supports
- a data structure that supports arbitary types (including complex, half, and arbitary-precision)
- a framework that supports different backends
- fast, at least efficient on server CPU
- supports parallel by threading (specifically rayon)
- large dynamic dimension tensor and its reshape
- functionality can be extended by other crates

And further more,
- the framework may not overwhelm chemist scientists

Many crates in native rust done well in some aspects but not all.

This crate gets inspires from [NumPy](https://github.com/data-apis/array-api/), [Array API standard](https://github.com/data-apis/array-api/), [ndarray](https://github.com/rust-ndarray/ndarray/), [candle](https://github.com/huggingface/candle), [Burn](https://github.com/tracel-ai/burn).

> **What is supposed to be supported in near future?**

- Lapack functions and basic linear algebra APIs
- CUDA and HIP device support
- MKL and BLIS device support
- Full support of [Python array API standard](https://data-apis.org/array-api/latest/) (in native rust instead of python binding)
    - statistical (reduction) functions (norm, std, etc)
    - searching functions
    - manuplication functions (stack, unstack, tile, roll, moveaxis)
- Einstein summation

> **What's RSTSR meaning?**

RSTSR actually refers to its relationship with **R**E**S**T **T**en**s**o**r** ([REST](https://github.com/igor-1982/rest)), instead of **R**u**s**t **T**en**s**o**r**. This crate was originally tried to developed a more dev-friendly experience for chemist programmer from numpy/scipy/pytorch.

> **Is there an illustrative project for using RSTSR in real-world project?**

We refer a project that developed before rstsr v0.1: [showcase of RI-CCSD](https://github.com/ajz34/showcase_rust_riccsd).
File [riccsd.rs](https://github.com/ajz34/showcase_rust_riccsd/blob/master/src/riccsd.rs) is a demonstration of code style to use RSTSR.

> **What features will not be implemented?**

We do not support autodiff and lazy-evaluation in far future. In this mean time, we are not very concern on machine-learning applications, but focus more on traditional scientific computing, especially applications in electronic structure.

## Miscellaneous

You are welcomed to raise problems or suggestions in github repo issues or discussions.

This project is still in early stage, and radical code factorization could occur; dev-documentation can still be greatly improved.
