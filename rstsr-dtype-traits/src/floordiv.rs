use num::{Float, Integer};

#[cfg(feature = "half")]
use half::{bf16, f16};

pub trait FloorDivideAPI {
    fn floor_divide(self, other: Self) -> Self;
}

macro_rules! impl_floordiv_int {
    ($t: ident) => {
        impl FloorDivideAPI for $t {
            fn floor_divide(self, other: Self) -> Self {
                Integer::div_floor(&self, &other)
            }
        }
    };

    ($T: ident, $($Ts: ident),*) => {
        impl_floordiv_int!($T);
        impl_floordiv_int!($($Ts),*);
    };
}

impl_floordiv_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! impl_floordiv_float {
    ($t: ident) => {
        impl FloorDivideAPI for $t {
            fn floor_divide(self, other: Self) -> Self {
                Float::floor(self / other)
            }
        }
    };

    ($T: ident, $($Ts: ident),*) => {
        impl_floordiv_float!($T);
        impl_floordiv_float!($($Ts),*);
    };
}

impl_floordiv_float!(f32, f64);
#[cfg(feature = "half")]
impl_floordiv_float!(bf16, f16);
