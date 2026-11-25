//! LAPACK interface layer (ADR-0010) of the non-symmetric eigendecomposition
//! drivers `?GEEV` (standard) and `?GGEV` (generalized `A v = lambda B v`):
//! the `XxxDriverAPI<T>` traits, the builder structs `GEEV_`/`GGEV_`, and the
//! aliases `S/D/C/ZGEEV`, `S/D/C/ZGGEV`.

use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub mod geev;
pub mod ggev;

pub use geev::*;
pub use ggev::*;

/// Eigenvalues of one 2-D matrix (the assembled complex spectrum).
pub type ComplexEigvals<B, T> = Tensor<Complex<<T as ComplexFloat>::Real>, B, Ix1>;

/// One block of complex eigenvectors (left or right), `(N, N)`.
pub type ComplexEigvecs<B, T> = Tensor<Complex<<T as ComplexFloat>::Real>, B, Ix2>;
