pub(crate) use rstsr_common::prelude_dev::*;
pub(crate) use rstsr_dtype_traits::DTypeCastAPI;

#[cfg(feature = "rayon")]
pub use rayon::ThreadPool;

/// Commonly used by the rayon kernels (parallel region pointer hoisting and
/// parallel iteration traits); see `cpu_rayon` module documentation.
#[cfg(feature = "rayon")]
pub use core::sync::atomic::{AtomicPtr, Ordering};
#[cfg(feature = "rayon")]
pub use rayon::prelude::*;

pub use crate::cpu_serial::adv_indexing::*;
pub use crate::cpu_serial::assignment::*;
pub use crate::cpu_serial::creation::*;
pub use crate::cpu_serial::matmul_naive::*;
pub use crate::cpu_serial::op_tri::*;
pub use crate::cpu_serial::op_with_func::*;
pub use crate::cpu_serial::reduction::*;
pub use crate::cpu_serial::transpose::*;
pub use crate::cpu_serial::vecdot::*;

#[cfg(feature = "rayon")]
mod cpu_rayon {
    pub use crate::cpu_rayon::adv_indexing::*;
    pub use crate::cpu_rayon::assignment::*;
    pub use crate::cpu_rayon::creation::*;
    pub use crate::cpu_rayon::matmul_naive::*;
    pub use crate::cpu_rayon::op_tri::*;
    pub use crate::cpu_rayon::op_with_func::*;
    pub use crate::cpu_rayon::reduction::*;
    pub use crate::cpu_rayon::transpose::*;
    pub use crate::cpu_rayon::vecdot::*;
}
#[cfg(feature = "rayon")]
pub use cpu_rayon::*;
