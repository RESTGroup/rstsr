pub(crate) use rstsr_common::prelude_dev::*;

pub use crate::cpu_serial::adv_indexing::*;
pub use crate::cpu_serial::assignment::*;
pub use crate::cpu_serial::matmul_naive::*;
pub use crate::cpu_serial::op_tri::*;
pub use crate::cpu_serial::op_with_func::*;
pub use crate::cpu_serial::reduction::*;
pub use crate::cpu_serial::transpose::*;

#[cfg(feature = "rayon")]
mod cpu_rayon {
    pub use crate::cpu_rayon::assignment::*;
    pub use crate::cpu_rayon::matmul_naive::*;
    pub use crate::cpu_rayon::op_tri::*;
    pub use crate::cpu_rayon::op_with_func::*;
    pub use crate::cpu_rayon::reduction::*;
    pub use crate::cpu_rayon::transpose::*;
}
#[cfg(feature = "rayon")]
pub use cpu_rayon::*;
