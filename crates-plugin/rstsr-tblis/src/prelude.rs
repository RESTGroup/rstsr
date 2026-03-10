pub mod rstsr_traits {
    pub use crate::einsum_impl::RTToTblisTensorAPI;
}

pub mod rstsr_funcs {
    pub use crate::einsum_impl::{einsum, einsum_f, einsum_with_output, einsum_with_output_f};
    pub use crate::tensordot_impl::{tensordot, tensordot_f};
}
