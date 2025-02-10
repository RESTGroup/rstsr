# RSTSR core crate

`rstsr-core` is the core of RSTSR series:
- It defines data structure and some traits (interface) of tensor, storage, device.
- It realizes two basic devices: [`DeviceCpuSerial`] and [`DeviceFaer`], so `rstsr-core` alone is a functional tensor toolkit library.

If you are more aware of matmul efficiency, or by other considerations (we will try to implement BLAS and Lapack features in future), you may find `DeviceOpenBLAS` in `rstsr-openblas` helpful. We hope to implement more devices in future.

## User document

Except for API document, we also tries to provide user document, refer to [readthedocs](https://rstsr-book.readthedocs.io/).

This document is still in construction.

Some information of developer guide will also shipped to that document.
