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
  - The `openmp` feature is a default feature (since 0.8.0), so an OpenMP runtime must also be linked: `gomp` on Linux (GNU toolchain), `omp` on macOS (LLVM), `vcomp` on Windows (MSVC).
    Please add `-l gomp` or `-l omp` in `RUSTFLAGS`, or `cargo:rustc-link-lib=gomp` or `cargo:rustc-link-lib=omp` in build.rs, or something similar, to your project.
    We do not use external FFI crate `openmp-sys`, and do not automatically search for OpenMP library for linking.

- Why `openmp` by default, and what it costs you:
  - In our testing, OpenBLAS compiled with OpenMP is generally more efficient than the pthread build, and OpenBLAS binaries built either way are common in the wild; with `openmp` enabled the crate handles both at runtime.
  - The feature is **compatible with a pthread-built OpenBLAS**: the runtime reports `OPENBLAS_THREAD` and the pthread API (`openblas_set_num_threads`) is used; the linked OpenMP runtime simply stays unused.
  - The only cost is the link requirement above. If you do not want it, disable default features and re-enable what you need:

    ```toml
    rstsr-openblas = { version = "0.8", default-features = false, features = ["linalg"] }
    ```

    Note that cargo feature unification is graph-global: if any other crate in your build enables `rstsr-openblas` default features, `openmp` is on for the whole build, and your opt-out alone will not remove the link requirement.
  - With `openmp` (and `dynamic_loading`) disabled, an OpenMP-built OpenBLAS makes the threading API panic with migration guidance. If you cannot link an OpenMP runtime at all, either use a pthread-built OpenBLAS, or enable `dynamic_loading` and resolve everything at runtime.

- Compatibility matrix (verified against OpenBLAS 0.3.34 built both ways, Linux/GNU toolchain):

| OpenBLAS build | `openmp` feature | OpenMP runtime linked | result |
| --- | --- | --- | --- |
| OpenMP | on (default) | yes | works; threading via `omp_*` |
| OpenMP | on | no | link error (`omp_*` undefined; OpenBLAS's own dependency on libgomp does not resolve them for you) |
| OpenMP | off | no | links; threading API panics with guidance |
| pthread | on (default) | yes | works; threading via `openblas_*`, OpenMP runtime unused |
| pthread | on | no | link error (`omp_*` undefined) |
| pthread | off | no | works; no OpenMP runtime needed |