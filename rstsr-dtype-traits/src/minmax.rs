use std::cmp::Ord;

#[cfg(feature = "half")]
use half::{bf16, f16};

pub trait MinMaxAPI {
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_minmax_direct_func {
    ($t: ident) => {
        impl MinMaxAPI for $t {
            fn min(self, other: Self) -> Self {
                $t::min(self, other)
            }

            fn max(self, other: Self) -> Self {
                $t::max(self, other)
            }
        }
    };

    ($T: ident, $($Ts: ident),*) => {
        impl_minmax_direct_func!($T);
        impl_minmax_direct_func!($($Ts),*);
    };
}

impl_minmax_direct_func!(f32, f64);
#[cfg(feature = "half")]
impl_minmax_direct_func!(bf16, f16);

macro_rules! impl_minmax_ord {
    ($t: ident) => {
        impl MinMaxAPI for $t {
            fn min(self, other: Self) -> Self {
                Ord::min(self, other)
            }

            fn max(self, other: Self) -> Self {
                Ord::max(self, other)
            }
        }
    };

    ($T: ident, $($Ts: ident),*) => {
        impl_minmax_ord!($T);
        impl_minmax_ord!($($Ts),*);
    };
}

impl_minmax_ord!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
