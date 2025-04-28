pub(crate) use rstsr_common::prelude_dev::*;

pub use crate::cpu_serial::assignment::*;
pub use crate::cpu_serial::matmul_naive::*;
pub use crate::cpu_serial::op_tri::*;
pub use crate::cpu_serial::op_with_func::*;
pub use crate::cpu_serial::reduction::*;

#[cfg(feature = "rayon")]
mod cpu_rayon {
    pub use crate::cpu_rayon::assignment::*;
    pub use crate::cpu_rayon::matmul_naive::*;
    pub use crate::cpu_rayon::op_tri::*;
    pub use crate::cpu_rayon::op_with_func::*;
    pub use crate::cpu_rayon::reduction::*;
}
#[cfg(feature = "rayon")]
pub use cpu_rayon::*;
