# RSTSR OpenBLAS device

This crate enables OpenBLAS device.

For more information of OpenBLAS and its usage, we refer to [document of rstsr-openblas-ffi](https://docs.rs/rstsr-openblas-ffi/).

## Usage

```rust
use rstsr_core::prelude::*;
use rstsr_openblas::DeviceOpenBLAS;

// specify the number of threads of 16
let device = DeviceOpenBLAS::new(16);
// if you want to use the default number of threads, use the following line
// let device = DeviceOpenBLAS::default();

let a = rt::linspace((0.0, 1.0, 1048576, &device)).into_shape([16, 256, 256]);
let b = rt::linspace((1.0, 2.0, 1048576, &device)).into_shape([16, 256, 256]);

// by optimized BLAS, the following operation is very fast
let c = &a % &b;

// mean of all elements is also performed in parallel
let c_mean = c.mean_all();

println!("{:?}", c_mean);
assert!((c_mean - 213.2503660477036) < 1e-6);
```

## Important Notes

- We do not provide automatic linkage:
  - Please add `-l openblas` in `RUSTFLAGS`, or `cargo:rustc-link-lib=openblas` in build.rs, or something similar, to your project.
    We do not use external FFI crates `blas` or `blas-sys`, and do not automatically search OpenBLAS library for linking.
  - If feature `openmp` activated, please add `-l gomp` or `-l omp` in `RUSTFLAGS`, or `cargo:rustc-link-lib=gomp` or `cargo:rustc-link-lib=omp` in build.rs, or something similar, to your project.
    We do not use external FFI crate `openmp-sys`, and do not automatically search for OpenMP library for linking.

- If your OpenBLAS is compiled with OpenMP, please add `openmp` feature to either this crate or `rstsr-openblas-ffi`.
  - In our testing, OpenBLAS with OpenMP is probably more efficient than pthreads. However, we currently decided not make `openmp` as default feature.