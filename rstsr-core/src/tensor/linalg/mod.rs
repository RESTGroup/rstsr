pub mod matmul;
pub mod matrix_transpose;
pub mod vecdot;

pub mod exports {
    use super::*;

    pub use matmul::*;
    pub use matrix_transpose::*;
    pub use vecdot::*;
}
