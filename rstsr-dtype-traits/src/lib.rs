#![doc = include_str!("../readme.md")]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod ext_float;
mod ext_num;
mod ext_real;
mod isclose;
mod promotion;
mod scalar;
mod val_write;

pub use ext_float::*;
pub use ext_num::*;
pub use ext_real::*;
pub use isclose::*;
pub use promotion::*;
pub use scalar::*;
pub use val_write::*;
