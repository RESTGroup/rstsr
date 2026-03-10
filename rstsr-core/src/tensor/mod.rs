pub mod adv_indexing;
pub mod asarray;
pub mod assignment;
pub mod creation;
pub mod creation_from_tensor;
pub mod device_conversion;
pub mod ext_conversion;
pub mod indexing;
pub mod iterator_axes;
pub mod iterator_elem;
pub mod linalg;
pub mod manuplication;
pub mod map_elementwise;
pub mod operators;
pub mod ownership_conversion;
pub mod pack_array;
pub mod reduction;
pub mod tensor2_impl;
pub mod tensor_mutable;
pub mod tensor_view_list;

#[allow(unused_imports)]
pub mod exports {
    use super::*;

    pub use adv_indexing::*;
    pub use asarray::*;
    pub use assignment::*;
    pub use creation::*;
    pub use creation_from_tensor::*;
    pub use device_conversion::*;
    pub use ext_conversion::*;
    pub use indexing::*;
    pub use iterator_axes::*;
    pub use iterator_elem::*;
    pub use linalg::exports::*;
    pub use manuplication::exports::*;
    pub use map_elementwise::*;
    pub use operators::exports::*;
    pub use ownership_conversion::*;
    pub use pack_array::*;
    pub use reduction::*;
    pub use tensor2_impl::*;
    pub use tensor_mutable::*;
    pub use tensor_view_list::*;
}
