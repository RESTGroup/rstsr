//! Data type promotion traits.
//!
//! This follows NumPy's convention:
//! <https://numpy.org/doc/stable/reference/arrays.promotion.html>

#![allow(non_camel_case_types)]

/* #region trait definition and basic implementation */

type c32 = num::complex::Complex<f32>;
type c64 = num::complex::Complex<f64>;
use duplicate::duplicate_item;
use num::Complex;
use num::{One, Zero};

pub trait PromotionSpecialAPI {
    type FloatType;
    type SumType;
    fn to_float_type(self) -> Self::FloatType;
    fn to_sum_type(self) -> Self::SumType;
}

pub trait PromotionAPI<T: Clone> {
    type Res;
    const SAME_TYPE: bool = false;
    const CAN_CAST_SELF: bool = false;
    const CAN_CAST_OTHER: bool = false;
    const CAN_ASTYPE: bool = false;
    fn promote_self(self) -> Self::Res;
    fn promote_other(val: T) -> Self::Res;
    fn promote_astype(self) -> T;
    #[inline]
    fn promote_pair(self, val: T) -> (Self::Res, Self::Res)
    where
        Self: Sized,
    {
        (self.promote_self(), Self::promote_other(val))
    }
}

impl<T: Clone> PromotionAPI<T> for T {
    type Res = T;
    const SAME_TYPE: bool = true;
    const CAN_CAST_SELF: bool = true;
    const CAN_CAST_OTHER: bool = true;
    const CAN_ASTYPE: bool = true;
    #[inline]
    fn promote_self(self) -> Self::Res {
        self
    }
    #[inline]
    fn promote_other(val: T) -> Self::Res {
        val
    }
    #[inline]
    fn promote_astype(self) -> T {
        self
    }
}

pub trait PromotionValAPI {
    fn promote<P: Clone>(self) -> Self::Res
    where
        Self: PromotionAPI<P>;
    fn astype<P: Clone>(self) -> P
    where
        Self: PromotionAPI<P>;
}

impl<T> PromotionValAPI for T
where
    T: Clone,
{
    fn promote<P: Clone>(self) -> <T as PromotionAPI<P>>::Res
    where
        Self: PromotionAPI<P>,
    {
        self.promote_self()
    }

    fn astype<P: Clone>(self) -> P
    where
        Self: PromotionAPI<P>,
    {
        self.promote_astype()
    }
}

/* #endregion */

/* #region PromotionSpecialAPI */

#[duplicate_item(T; [u8]; [u16]; [u32]; [u64];)]
impl PromotionSpecialAPI for T {
    type FloatType = f64;
    type SumType = u64;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self as _
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self as _
    }
}

#[duplicate_item(T; [i8]; [i16]; [i32]; [i64];)]
impl PromotionSpecialAPI for T {
    type FloatType = f64;
    type SumType = i64;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self as _
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self as _
    }
}

#[duplicate_item(T; [f32]; [f64]; [c32]; [c64];)]
impl PromotionSpecialAPI for T {
    type FloatType = T;
    type SumType = T;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl PromotionSpecialAPI for T {
    type FloatType = T;
    type SumType = T;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self
    }
}

impl PromotionSpecialAPI for usize {
    type FloatType = f64;
    type SumType = usize;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self as _
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self
    }
}

impl PromotionSpecialAPI for isize {
    type FloatType = f64;
    type SumType = isize;
    #[inline]
    fn to_float_type(self) -> Self::FloatType {
        self as _
    }
    #[inline]
    fn to_sum_type(self) -> Self::SumType {
        self
    }
}

/* #endregion */

/* #region rule bool<T> */

macro_rules! impl_promotion_bool_T {
    ($T:ty) => {
        impl PromotionAPI<$T> for bool {
            type Res = $T;
            const CAN_CAST_OTHER: bool = true;
            const CAN_ASTYPE: bool = true;
            #[inline]
            fn promote_self(self) -> Self::Res {
                if self {
                    <$T>::one()
                } else {
                    <$T>::zero()
                }
            }
            #[inline]
            fn promote_other(val: $T) -> Self::Res {
                val
            }
            #[inline]
            fn promote_astype(self) -> $T {
                if self {
                    <$T>::one()
                } else {
                    <$T>::zero()
                }
            }
        }

        impl PromotionAPI<bool> for $T {
            type Res = $T;
            const CAN_CAST_SELF: bool = true;
            const CAN_ASTYPE: bool = true;
            #[inline]
            fn promote_self(self) -> Self::Res {
                self
            }
            #[inline]
            fn promote_other(val: bool) -> Self::Res {
                if val {
                    <$T>::one()
                } else {
                    <$T>::zero()
                }
            }
            #[inline]
            fn promote_astype(self) -> bool {
                self != <$T>::zero()
            }
        }
    };
}

// internal type
impl_promotion_bool_T!(u8);
impl_promotion_bool_T!(u16);
impl_promotion_bool_T!(u32);
impl_promotion_bool_T!(u64);
impl_promotion_bool_T!(i8);
impl_promotion_bool_T!(i16);
impl_promotion_bool_T!(i32);
impl_promotion_bool_T!(i64);
impl_promotion_bool_T!(f32);
impl_promotion_bool_T!(f64);
// external type
impl_promotion_bool_T!(usize);
impl_promotion_bool_T!(isize);
#[cfg(feature = "half")]
impl_promotion_bool_T!(half::f16);
#[cfg(feature = "half")]
impl_promotion_bool_T!(half::bf16);
// complex float
impl_promotion_bool_T!(c32);
impl_promotion_bool_T!(c64);

/* #endregion */

/* #region as-able primitive types */

macro_rules! impl_promotion_asable {
    ($T1:ty, $T2:ty, $can_cast_self: ident, $can_cast_other: ident, $Res:ty) => {
        impl PromotionAPI<$T2> for $T1 {
            type Res = $Res;
            const CAN_CAST_SELF: bool = $can_cast_self;
            const CAN_CAST_OTHER: bool = $can_cast_other;
            const CAN_ASTYPE: bool = true;
            #[inline]
            fn promote_self(self) -> Self::Res {
                self as $Res
            }
            #[inline]
            fn promote_other(val: $T2) -> Self::Res {
                val as $Res
            }
            #[inline]
            fn promote_astype(self) -> $T2 {
                self as $T2
            }
        }
    };
}

// internal typeimpl_promotion_asable!(i8, i16, false, true, i16);
impl_promotion_asable!(i8, i32, false, true, i32);
impl_promotion_asable!(i8, i64, false, true, i64);
impl_promotion_asable!(i8, u8, false, false, i16);
impl_promotion_asable!(i8, u16, false, false, i32);
impl_promotion_asable!(i8, u32, false, false, i64);
impl_promotion_asable!(i8, u64, false, false, f64);
impl_promotion_asable!(i8, f32, false, true, f32);
impl_promotion_asable!(i8, f64, false, true, f64);
impl_promotion_asable!(i16, i8, true, false, i16);
impl_promotion_asable!(i16, i32, false, true, i32);
impl_promotion_asable!(i16, i64, false, true, i64);
impl_promotion_asable!(i16, u8, true, false, i16);
impl_promotion_asable!(i16, u16, false, false, i32);
impl_promotion_asable!(i16, u32, false, false, i64);
impl_promotion_asable!(i16, u64, false, false, f64);
impl_promotion_asable!(i16, f32, false, true, f32);
impl_promotion_asable!(i16, f64, false, true, f64);
impl_promotion_asable!(i32, i8, true, false, i32);
impl_promotion_asable!(i32, i16, true, false, i32);
impl_promotion_asable!(i32, i64, false, true, i64);
impl_promotion_asable!(i32, u8, true, false, i32);
impl_promotion_asable!(i32, u16, true, false, i32);
impl_promotion_asable!(i32, u32, false, false, i64);
impl_promotion_asable!(i32, u64, false, false, f64);
impl_promotion_asable!(i32, f32, false, false, f64);
impl_promotion_asable!(i32, f64, false, true, f64);
impl_promotion_asable!(i64, i8, true, false, i64);
impl_promotion_asable!(i64, i16, true, false, i64);
impl_promotion_asable!(i64, i32, true, false, i64);
impl_promotion_asable!(i64, u8, true, false, i64);
impl_promotion_asable!(i64, u16, true, false, i64);
impl_promotion_asable!(i64, u32, true, false, i64);
impl_promotion_asable!(i64, u64, false, false, f64);
impl_promotion_asable!(i64, f32, false, false, f64);
impl_promotion_asable!(i64, f64, false, true, f64);
impl_promotion_asable!(u8, i8, false, false, i16);
impl_promotion_asable!(u8, i16, false, true, i16);
impl_promotion_asable!(u8, i32, false, true, i32);
impl_promotion_asable!(u8, i64, false, true, i64);
impl_promotion_asable!(u8, u16, false, true, u16);
impl_promotion_asable!(u8, u32, false, true, u32);
impl_promotion_asable!(u8, u64, false, true, u64);
impl_promotion_asable!(u8, f32, false, true, f32);
impl_promotion_asable!(u8, f64, false, true, f64);
impl_promotion_asable!(u16, i8, false, false, i32);
impl_promotion_asable!(u16, i16, false, false, i32);
impl_promotion_asable!(u16, i32, false, true, i32);
impl_promotion_asable!(u16, i64, false, true, i64);
impl_promotion_asable!(u16, u8, true, false, u16);
impl_promotion_asable!(u16, u32, false, true, u32);
impl_promotion_asable!(u16, u64, false, true, u64);
impl_promotion_asable!(u16, f32, false, true, f32);
impl_promotion_asable!(u16, f64, false, true, f64);
impl_promotion_asable!(u32, i8, false, false, i64);
impl_promotion_asable!(u32, i16, false, false, i64);
impl_promotion_asable!(u32, i32, false, false, i64);
impl_promotion_asable!(u32, i64, false, true, i64);
impl_promotion_asable!(u32, u8, true, false, u32);
impl_promotion_asable!(u32, u16, true, false, u32);
impl_promotion_asable!(u32, u64, false, true, u64);
impl_promotion_asable!(u32, f32, false, false, f64);
impl_promotion_asable!(u32, f64, false, true, f64);
impl_promotion_asable!(u64, i8, false, false, f64);
impl_promotion_asable!(u64, i16, false, false, f64);
impl_promotion_asable!(u64, i32, false, false, f64);
impl_promotion_asable!(u64, i64, false, false, f64);
impl_promotion_asable!(u64, u8, true, false, u64);
impl_promotion_asable!(u64, u16, true, false, u64);
impl_promotion_asable!(u64, u32, true, false, u64);
impl_promotion_asable!(u64, f32, false, false, f64);
impl_promotion_asable!(u64, f64, false, true, f64);
impl_promotion_asable!(f32, i8, true, false, f32);
impl_promotion_asable!(f32, i16, true, false, f32);
impl_promotion_asable!(f32, i32, false, false, f64);
impl_promotion_asable!(f32, i64, false, false, f64);
impl_promotion_asable!(f32, u8, true, false, f32);
impl_promotion_asable!(f32, u16, true, false, f32);
impl_promotion_asable!(f32, u32, false, false, f64);
impl_promotion_asable!(f32, u64, false, false, f64);
impl_promotion_asable!(f32, f64, false, true, f64);
impl_promotion_asable!(f64, i8, true, false, f64);
impl_promotion_asable!(f64, i16, true, false, f64);
impl_promotion_asable!(f64, i32, true, false, f64);
impl_promotion_asable!(f64, i64, true, false, f64);
impl_promotion_asable!(f64, u8, true, false, f64);
impl_promotion_asable!(f64, u16, true, false, f64);
impl_promotion_asable!(f64, u32, true, false, f64);
impl_promotion_asable!(f64, u64, true, false, f64);
impl_promotion_asable!(f64, f32, true, false, f64);

// external type: isize
impl_promotion_asable!(isize, i8, true, false, isize);
impl_promotion_asable!(isize, i16, true, false, isize);
impl_promotion_asable!(isize, i32, true, false, isize);
impl_promotion_asable!(isize, i64, true, true, isize);
impl_promotion_asable!(isize, u8, true, false, isize);
impl_promotion_asable!(isize, u16, true, false, isize);
impl_promotion_asable!(isize, u32, true, false, isize);
impl_promotion_asable!(isize, u64, false, false, f64);
impl_promotion_asable!(isize, f32, false, false, f64);
impl_promotion_asable!(isize, f64, false, true, f64);
impl_promotion_asable!(i8, isize, false, true, isize);
impl_promotion_asable!(i16, isize, false, true, isize);
impl_promotion_asable!(i32, isize, false, true, isize);
impl_promotion_asable!(i64, isize, true, true, isize);
impl_promotion_asable!(u8, isize, false, true, isize);
impl_promotion_asable!(u16, isize, false, true, isize);
impl_promotion_asable!(u32, isize, false, true, isize);
impl_promotion_asable!(u64, isize, false, false, f64);
impl_promotion_asable!(f32, isize, false, false, f64);
impl_promotion_asable!(f64, isize, true, false, f64);

// external type: usize
impl_promotion_asable!(usize, i8, false, false, f64);
impl_promotion_asable!(usize, i16, false, false, f64);
impl_promotion_asable!(usize, i32, false, false, f64);
impl_promotion_asable!(usize, i64, false, false, f64);
impl_promotion_asable!(usize, u8, true, false, usize);
impl_promotion_asable!(usize, u16, true, false, usize);
impl_promotion_asable!(usize, u32, true, false, usize);
impl_promotion_asable!(usize, u64, true, true, usize);
impl_promotion_asable!(usize, f32, false, false, f64);
impl_promotion_asable!(usize, f64, false, true, f64);
impl_promotion_asable!(i8, usize, false, false, f64);
impl_promotion_asable!(i16, usize, false, false, f64);
impl_promotion_asable!(i32, usize, false, false, f64);
impl_promotion_asable!(i64, usize, false, false, f64);
impl_promotion_asable!(u8, usize, false, true, usize);
impl_promotion_asable!(u16, usize, false, true, usize);
impl_promotion_asable!(u32, usize, false, true, usize);
impl_promotion_asable!(u64, usize, true, true, usize);
impl_promotion_asable!(f32, usize, false, false, f64);
impl_promotion_asable!(f64, usize, true, false, f64);

/* #endregion */

/* #region complex to primitive */

macro_rules! impl_promotion_complex_primitive_cast_self {
    ($TComp:ty, $TPrim:ty, $can_cast_self:ident, $can_cast_other:ident, $ResComp:ty) => {
        impl PromotionAPI<$TPrim> for Complex<$TComp> {
            type Res = Complex<$ResComp>;
            const CAN_CAST_SELF: bool = $can_cast_self;
            const CAN_CAST_OTHER: bool = $can_cast_other;
            const CAN_ASTYPE: bool = false;
            #[inline]
            fn promote_self(self) -> Self::Res {
                self
            }
            #[inline]
            fn promote_other(val: $TPrim) -> Self::Res {
                Self::Res::new(val as _, 0 as _)
            }
            #[inline]
            fn promote_astype(self) -> $TPrim {
                panic!("Cannot cast complex to primitive type.");
            }
        }
    };
}

macro_rules! impl_promotion_complex_primitive_no_cast_self {
    ($TComp:ty, $TPrim:ty, $can_cast_self:ident, $can_cast_other:ident, $ResComp:ty) => {
        impl PromotionAPI<$TPrim> for Complex<$TComp> {
            type Res = Complex<$ResComp>;
            const CAN_CAST_SELF: bool = $can_cast_self;
            const CAN_CAST_OTHER: bool = $can_cast_other;
            const CAN_ASTYPE: bool = false;
            #[inline]
            fn promote_self(self) -> Self::Res {
                Self::Res::new(self.re as _, self.im as _)
            }
            #[inline]
            fn promote_other(val: $TPrim) -> Self::Res {
                Self::Res::new(val as _, 0 as _)
            }
            #[inline]
            fn promote_astype(self) -> $TPrim {
                panic!("Cannot cast complex to primitive type.");
            }
        }
    };
}

impl_promotion_complex_primitive_cast_self!(f32, i8, true, false, f32);
impl_promotion_complex_primitive_cast_self!(f32, i16, true, false, f32);
impl_promotion_complex_primitive_cast_self!(f32, u8, true, false, f32);
impl_promotion_complex_primitive_cast_self!(f32, u16, true, false, f32);
impl_promotion_complex_primitive_cast_self!(f32, f32, true, false, f32);

impl_promotion_complex_primitive_cast_self!(f64, i8, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, i16, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, i32, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, i64, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, isize, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, u8, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, u16, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, u32, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, u64, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, usize, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, f32, true, false, f64);
impl_promotion_complex_primitive_cast_self!(f64, f64, true, false, f64);

impl_promotion_complex_primitive_no_cast_self!(f32, i32, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, i64, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, isize, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, u32, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, u64, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, usize, false, false, f64);
impl_promotion_complex_primitive_no_cast_self!(f32, f64, false, false, f64);

/* #endregion */

/* #region primitive to complex */

macro_rules! impl_promotion_primitive_complex_cast_other {
    ($TComp:ty, $TPrim:ty, $can_cast_self:ident, $can_cast_other:ident, $ResComp:ty) => {
        impl PromotionAPI<Complex<$TComp>> for $TPrim {
            type Res = Complex<$ResComp>;
            const CAN_CAST_SELF: bool = $can_cast_self;
            const CAN_CAST_OTHER: bool = $can_cast_other;
            const CAN_ASTYPE: bool = true;
            #[inline]
            fn promote_self(self) -> Self::Res {
                Self::Res::new(self as _, 0 as _)
            }
            #[inline]
            fn promote_other(val: Complex<$TComp>) -> Self::Res {
                val
            }
            #[inline]
            fn promote_astype(self) -> Complex<$TComp> {
                Complex::<$TComp>::new(self as _, 0 as _)
            }
        }
    };
}

macro_rules! impl_promotion_primitive_complex_nocast_other {
    ($TComp:ty, $TPrim:ty, $can_cast_self:ident, $can_cast_other:ident, $ResComp:ty) => {
        impl PromotionAPI<Complex<$TComp>> for $TPrim {
            type Res = Complex<$ResComp>;
            const CAN_CAST_SELF: bool = $can_cast_self;
            const CAN_CAST_OTHER: bool = $can_cast_other;
            const CAN_ASTYPE: bool = true;
            #[inline]
            fn promote_self(self) -> Self::Res {
                Self::Res::new(self as _, 0 as _)
            }
            #[inline]
            fn promote_other(val: Complex<$TComp>) -> Self::Res {
                Self::Res::new(val.re as _, val.im as _)
            }
            #[inline]
            fn promote_astype(self) -> Complex<$TComp> {
                Complex::<$TComp>::new(self as _, 0 as _)
            }
        }
    };
}

impl_promotion_primitive_complex_cast_other!(f32, i8, false, true, f32);
impl_promotion_primitive_complex_cast_other!(f32, i16, false, true, f32);
impl_promotion_primitive_complex_cast_other!(f32, u8, false, true, f32);
impl_promotion_primitive_complex_cast_other!(f32, u16, false, true, f32);
impl_promotion_primitive_complex_cast_other!(f32, f32, false, true, f32);

impl_promotion_primitive_complex_nocast_other!(f64, i8, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, i16, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, i32, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, i64, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, isize, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, u8, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, u16, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, u32, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, u64, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, usize, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, f32, false, true, f64);
impl_promotion_primitive_complex_nocast_other!(f64, f64, false, true, f64);

impl_promotion_primitive_complex_nocast_other!(f32, i32, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, i64, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, isize, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, u32, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, u64, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, usize, false, false, f64);
impl_promotion_primitive_complex_nocast_other!(f32, f64, false, false, f64);

/* #endregion */

/* #region complex to complex */

impl PromotionAPI<c32> for c64 {
    type Res = c64;
    const CAN_CAST_SELF: bool = true;
    const CAN_CAST_OTHER: bool = false;
    const CAN_ASTYPE: bool = true;
    #[inline]
    fn promote_self(self) -> Self::Res {
        self
    }
    #[inline]
    fn promote_other(val: c32) -> Self::Res {
        c64::new(val.re as f64, val.im as f64)
    }
    #[inline]
    fn promote_astype(self) -> c32 {
        c32::new(self.re as f32, self.im as f32)
    }
}

impl PromotionAPI<c64> for c32 {
    type Res = c64;
    const CAN_CAST_SELF: bool = false;
    const CAN_CAST_OTHER: bool = true;
    const CAN_ASTYPE: bool = true;
    #[inline]
    fn promote_self(self) -> Self::Res {
        c64::new(self.re as f64, self.im as f64)
    }
    #[inline]
    fn promote_other(val: c64) -> Self::Res {
        val
    }
    #[inline]
    fn promote_astype(self) -> c64 {
        c64::new(self.re as f64, self.im as f64)
    }
}

/* #endregion */
