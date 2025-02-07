# RSTSR sub-project OpenBLAS-FFI

This sub-project is used for generating BLAS FFI for OpenBLAS.

There are already some rust BLAS FFI crates. However, there may be some additional issues to be tackled:
- LP64/ILP64 support,
- Custom suffixes,
- Non-standard BLAS functions,
- OpenBLAS specific utilities,
- For OpenMP or pthread compiled OpenBLAS, we need to add some utilities to both handling parallel and single-threaded.

Not all tasks will be handled currently. We will first try to make OpenBLAS device in RSTSR work.
