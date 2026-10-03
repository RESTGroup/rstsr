# rstsr-cpu-dlpack

DLPack interchange for rstsr CPU tensors: convert between rstsr tensors and
[DLPack](https://github.com/dmlc/dlpack) managed tensors — the mechanism
NumPy/PyTorch/CuPy and others use to exchange tensors with foreign libraries.

The crate is pure Rust (no Python bindings): it works with raw
`DLManagedTensorVersioned` / `DLManagedTensor` pointers, so it can serve any
DLPack consumer — another Rust crate, C, Julia — not only NumPy. Python
exchange is a thin host-side capsule holder over this crate.

- **export**: owned tensors (ownership transfer), shared tensors (zero-copy,
  read-only) and a copy fallback for views.
- **import**: zero-copy read-only tensors over foreign buffers; the imported
  tensor owns the producer's lifetime (the DLPack deleter travels with it).
- DLPack 1.0 interchange subset (`version = {1, 0}`); all CPU devices
  (`Raw = Vec<T>`) map to `kDLCPU`.
- **cargo features**: `half` (default; `f16`/`bf16` dtype support, via the
  `half` crate), `row_major` / `col_major` and `std` (forwarded to
  `rstsr-core` and `rstsr-common`).
