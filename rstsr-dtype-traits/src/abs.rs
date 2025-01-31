use num::{Complex, Float};

#[cfg(feature = "half")]
use half::{bf16, f16};

pub trait AbsAPI {
    type Out;
    const UNCHANGED: bool;
    const SAME_TYPE: bool;
    fn abs(self) -> Self::Out;
}

macro_rules! impl_abs_ux {
    ($t: ident) => {
        impl AbsAPI for $t {
            type Out = $t;
            const UNCHANGED: bool = true;
            const SAME_TYPE: bool = true;

            fn abs(self) -> Self::Out {
                self
            }
        }
    };

    ($t: ident, $($ts: ident),*) => {
        impl_abs_ux!($t);
        impl_abs_ux!($($ts),*);
    };
}

impl_abs_ux!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_abs_signed {
    ($t: ident) => {
        impl AbsAPI for $t {
            type Out = $t;
            const UNCHANGED: bool = false;
            const SAME_TYPE: bool = true;

            fn abs(self) -> Self::Out {
                $t::abs(self)
            }
        }
    };

    ($t: ident, $($ts: ident),*) => {
        impl_abs_signed!($t);
        impl_abs_signed!($($ts),*);
    };
}

impl_abs_signed!(i8, i16, i32, i64, i128, isize);

macro_rules! impl_abs_float {
    ($t: ident) => {
        impl AbsAPI for $t {
            type Out = $t;
            const UNCHANGED: bool = false;
            const SAME_TYPE: bool = true;

            fn abs(self) -> Self::Out {
                Float::abs(self)
            }
        }
    };

    ($t: ident, $($ts: ident),*) => {
        impl_abs_float!($t);
        impl_abs_float!($($ts),*);
    };
}

impl_abs_float!(f32, f64);
#[cfg(feature = "half")]
impl_abs_float!(bf16, f16);

impl<T> AbsAPI for Complex<T>
where
    T: Float,
{
    type Out = T;
    const UNCHANGED: bool = false;
    const SAME_TYPE: bool = false;

    fn abs(self) -> Self::Out {
        self.norm()
    }
}
