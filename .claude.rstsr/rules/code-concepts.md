## Core Concepts

- **Tensor types**: `TensorAny<R, T, B, D>`: (representation/ownership, dtype, backend/device, dimensionality)
- **Ownership**:
  - `Tensor<T, B, D>`, `TensorView<'a, T, B, D>`, `TensorMut<'a, T, B, D>`, `TensorCow<'a, T, B, D>`, `TensorArc<T, B, D>`
  - `TensorBase<S, D>`: (storage, dimensionality)
- **Layout**:
  - `Layout<D>` where dimensionality represented by `[usize; N]` for static or `Vec<usize>` for dynamic.
  - contains shape (by list of `usize`), stride (by list of `isize`), and offset (by `usize`).
- **Manuplication**:
  - tensor composed as (storage = (repr/ownership, data), layout = (shape, stride, offset))
  - layout change: basic-indexing (slicing), transpose, permute, broadcast, etc.
    - `into_<func>`: `TensorAny` -> `TensorAny` (ownership preserved)
    - `to_<func>`: `&TensorAny` -> `TensorView` (only view, no copy)
  - conditional copy-on-write: reshape, etc.
    - `into_shape`: `TensorAny` -> `TensorCow` (returns owned)
    - `reshape`/`to_shape`: `&TensorAny` -> `TensorCow` (copy only if necessary, otherwise view)
    - `change_shape`: `TensorAny` -> `TensorCow` (less used)
  - explicit copy: advanced-indexing, etc.

## Code Style

- **Trait-based**: Operations are always defined as traits, and device backends implement these traits. For example `OpAddAPI`, `OpSinAPI`, `DeviceChangeAPI`.
- **Naming Convention**: (usually, may have exceptions)
  - traits add `API` suffix
  - device traits add `Device` prefix
  - error propagation function add `_f` suffix, eg `reshape_f` and return `Result<T>` (rstsr's custom result type)
  - panic version without `_f` suffix, eg `reshape`, note to use `rstsr_unwrap()` to allow backtrace
- **Trait-based Overload**:
  - `asarray` function as example (`rt::asarray((Vec<T>, &D)) -> Tensor`, `rt::asarray((&[T], &D)) -> TensorView`, etc)
  - `rt::add` function as example (`rt::add(&a, &b)` use views, `rt::add(a, &b)` use owned and may reuse storage of `a` if possible, etc)
- **Other Conventions**
  - prefer `foo(bar: impl Bar)` over `fn foo<T>(bar: T) where T: Bar` for if trait bound not complicated (no composition or multiple bounds)
  - prefer `impl<T> where T: trait` over `impl<T: trait>` for better readability, unless trait bound is extremely simple (like `Clone`, `Default`, etc)

## Common Commands

- test: example `cargo test --package rstsr-core --test tests_core_row --features backtrace --no-default-features -- tests_core::manuplication::test_reshape::numpy_reshape::regression --exact --nocapture`
  - use default if testing parallel
  - use `col_major` feature if testing column-major
  - note different crates (packages) have different features
- doc build: `RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps`
