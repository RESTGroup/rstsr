#[cfg(feature = "half")]
use half::{bf16, f16};
use num::{Complex, Float, Zero};

pub trait ReImAPI: Clone {
    type Out: Clone + ReImAPI<Out = Self::Out>;
    const REALIDENT: bool;
    const SAME_TYPE: bool;
    fn real(self) -> Self::Out;
    fn imag(self) -> Self::Out;
}

macro_rules! impl_unchanged {
    ($t: ident) => {
        impl ReImAPI for $t {
            type Out = $t;
            const REALIDENT: bool = true;
            const SAME_TYPE: bool = true;

            fn real(self) -> Self::Out {
                self
            }

            fn imag(self) -> Self::Out {
                $t::zero()
            }
        }
    };

    ($t: ident, $($ts: ident),*) => {
        impl_unchanged!($t);
        impl_unchanged!($($ts),*);
    };
}

impl_unchanged!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
#[cfg(feature = "half")]
impl_unchanged!(bf16, f16);

impl<T> ReImAPI for Complex<T>
where
    T: Float + ReImAPI<Out = T>,
{
    type Out = T;
    const REALIDENT: bool = false;
    const SAME_TYPE: bool = false;

    fn real(self) -> Self::Out {
        self.re
    }

    fn imag(self) -> Self::Out {
        self.im
    }
}
