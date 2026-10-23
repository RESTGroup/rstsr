//! Tensor functions organized by category: creation, indexing, manipulation,
//! element-wise operators, reduction, and linalg, plus ownership, device, and
//! external-type conversions.

pub mod adv_indexing;
pub mod array_indexer;
pub mod array_indexing;
pub mod asarray;
pub mod assignment;
pub mod astype;
pub mod creation;
pub mod creation_from_tensor;
pub mod device_conversion;
pub mod diff;
pub mod ext_conversion;
pub mod ext_linalg;
pub mod indexing;
pub mod iterator_axes;
pub mod iterator_elem;
pub mod linalg;
pub mod manipulation;
pub mod map_elementwise;
pub mod nonzero;
pub mod operators;
pub mod ownership_conversion;
pub mod pack_array;
pub mod reduction;
pub mod searching;
pub mod set;
pub mod sorting;
pub mod tensor2_impl;
pub mod tensor_mutable;
pub mod tensor_view_list;

#[allow(unused_imports)]
pub mod exports {
    use super::*;

    pub use adv_indexing::*;
    pub use array_indexer::*;
    pub use array_indexing::*;
    pub use asarray::*;
    pub use assignment::*;
    pub use astype::*;
    pub use creation::*;
    pub use creation_from_tensor::*;
    pub use device_conversion::*;
    pub use diff::*;
    pub use ext_conversion::*;
    pub use ext_linalg::exports::*;
    pub use indexing::*;
    pub use iterator_axes::*;
    pub use iterator_elem::*;
    pub use linalg::exports::*;
    pub use manipulation::exports::*;
    pub use map_elementwise::*;
    pub use nonzero::*;
    pub use operators::exports::*;
    pub use ownership_conversion::*;
    pub use pack_array::*;
    pub use reduction::*;
    pub use searching::*;
    pub use set::*;
    pub use sorting::*;
    pub use tensor2_impl::*;
    pub use tensor_mutable::*;
    pub use tensor_view_list::*;
}
