# RSTSR: A Rust Tensor Toolkit

**This is a crate in early development. It has not reached a version 0.1 but very near.**

**A possible roadmap listed below.**

## Current Features

- **Device Dependent:** This crate allows multiple devices/backends. By design, this crate separates tensor API and algorithm implementation API; all devices/backends share the same tensor API, while different in algorithm implementation API. Currently, `DeviceCpuSerial` (as reference implementation) and `DeviceFaer` is created. Other devices/backends could be implemented in future.
- **n-Dimensional and Broadcasting:** This crate provides n-dimensional tensor support, similar data-structure as ndarray in rust, and numpy in python. Broadcasting is available. We will try to implement as many [Python array API](https://data-apis.org/array-api/latest/) functions as we can. We also hope to make function APIs to be similar to that of python/numpy.
- **% (remainder) as Matmul:** This crate is radical in adopting matrix-multiplication operator. Remainder operator `%` is rarely used in floating-point arithmetics and integer matrix/vector computations. We use `%` as matrix-multiplication operator, like `@` in python [PEP-465](https://peps.python.org/pep-0465/).
    ```rust
    // `a` and `b` are tensor objects
    let c = &a % &b;  // perform `a.matmul(&b)`
    ```
- **Fast Matmul by Faer:** With `DeviceFaer`, efficiency of matrix-multiplication (of `f32`, `f64`, `Complex<f32>`, `Complex<f64>`) should be comparable to (some cases even faster than) highly-optimized BLAS. Further more, $C = A A^T$ can be further speeded-up by `SYRK`. Though matrix symmetrize is not fully optimized, current implementation in `DeviceFaer` will handle $C = A A^T$ faster than general `GEMM`, comparable to that of `numpy`.
- **Parallel in Complicated Layouts:** For example, in most cases, tensor addition $C = A + B$ is fast enough in serial (one-thread). Compiler nowadays usually automatically generates vectorized assemblies for this kind of task, even for naive implementation. However, when layout is not match (something like $C = A + B^T$), it can be extremely inefficient due to cache miss. `rstsr` does not prefectly solves this problem (decreases cache miss by tiling tensor), but try to perform tensor addition by parallel. Given enough threads, parallel can give 2--8 times efficiency boost.
- **Full Support of Basic Indexing:** *Basic* indexing is not essentially *basic*, but a scheme of layout manuplication. The tensor after basic indexing can still be represented as contiguous memory with strided layout. Few rust libraries fully supports tensor basic indexing including dimension selection, striding, addition (unsqueeze), ellipsis. RSTSR supports all of them. As a complicated example,
    ```rust
    use rstsr_core::prelude::*;
    let a: Tensor<f64, _> = rt::zeros([6, 5, 7, 4, 2]);
    let b = a.slice((slice!(-2, 1, -1), None, None, Ellipsis, 1, ..-1));
    println!("{:?}", b.layout());
    // output: shape: [3, 1, 1, 5, 7, 1], stride: [-280, 56, 56, 56, 8, 1], offset: 1122
    ```
- **Parallel Tensor Iterator:** Large tensors can be splitted into small tensors to be further parallel processed by threading. This is a feature can hardly be covered in python, and convenient to be realized in rust by rayon.

## Roadmap to version 0.1

Some requirements in [Python array API](https://data-apis.org/array-api/latest/)

- [x] basic struct and serial/parallel backends
- [x] broadcasting
- [x] creation functions
- [x] element-wise basic arithmetics (+, -, *, /, etc)
- [x] element-wise functions (sin, abs, floor, etc) and mapping by arbitary closures
- [x] statistical (reduction) function: sum
- [x] basic indexing
- [x] matmul
- [x] manuplication functions (partially done)

Other utilities

- [x] (parallel) index by axis
- [x] BLAS device: OpenBLAS

About crate

- [x] Minimal user document
- [x] Minimal correctness validation (testing)

## Roadmap for near future versions

Some requirements in [Python array API](https://data-apis.org/array-api/latest/)

- [ ] statistical (reduction) functions (norm, std, etc)
- [ ] searching functions
- [ ] manuplication functions (stack, unstack, tile, roll, moveaxis)

About crate

- [ ] User document
- [ ] Minimal dev document
- [ ] Correctness validation (testing)
- [ ] Coverage validation
- [ ] Github action

## Roadmap to future version

- BLAS (MKL, BLIS) device
- GPU device
- enhanced linalg
    - faster matrix congruence ($C = A^T B A$), which is used in ERI transformation
    - einops (may implement in another crate), which is convenient for many post-HF algorithms
- more numpy features and near-full support to Python array API
- user/dev/api documentation, testing, coverage, benchmarking
- user documentation for best/recommended practice
- optimization for memory-bounded operations in 1-D, 2-D cases (transpose, symmetrize, triangular-pack/unpack)
- more blas/lapack and linalg may be implemented in another crate, to support more requirements for chemistry applications

## Miscellaneous

`RSTSR` actually refers to its relationship with **R**E**S**T **T**en**s**o**r** ([REST](https://github.com/igor-1982/rest)), instead of **R**u**s**t **T**en**s**o**r**. This crate was originally tried to developed a more dev-friendly experience for chemist programmer from numpy/scipy/pytorch. But that can be a tough task.

It is grateful if you share your views on how to further improve this crate. This project is still in early stage, and radical code factorization could occur; dev-documentation has not been prepared now.

This crate gets inspires from `numpy`, `array-api`, `ndarray`, `candle`, `burn`.
