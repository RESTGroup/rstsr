#![allow(clippy::needless_return)]
#![allow(clippy::unnecessary_cast)]
#![doc = include_str!("../readme.md")]

pub mod cblas;
pub mod ffi;
pub mod threading;

pub use crate::threading::{get_num_threads, get_parallel, set_num_threads, with_num_threads};
