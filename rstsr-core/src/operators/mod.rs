//! Operations on tensors.

pub mod adv_indexing;
pub mod assignment;
pub mod combined_trait;
pub mod linalg;
pub mod matmul;
pub mod ops;
pub mod reduction;

pub mod exports {
    use super::*;

    pub use adv_indexing::*;
    pub use assignment::*;
    pub use combined_trait::*;
    pub use linalg::*;
    pub use matmul::*;
    pub use ops::*;
    pub use reduction::*;
}
