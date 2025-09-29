use duplicate::duplicate_item;
use num::{Float, Integer};

/// Extension trait for real types (floats and integers included).
pub trait ExtReal: Clone {
    /// Computes the floor division of two numbers.
    fn ext_floor_divide(self, other: Self) -> Self;

    /// Returns the minimum of two numbers.
    ///
    /// # Note
    ///
    /// For floats, this uses the `min` method which handles NaNs according to IEEE 754-2008 (the
    /// std library of rust).
    fn ext_min(self, other: Self) -> Self;

    /// The minimum value that can be represented by this type.
    fn ext_min_value() -> Self;

    /// Returns the maximum of two numbers.
    ///
    /// # Note
    ///
    /// For floats, this uses the `min` method which handles NaNs according to IEEE 754-2008 (the
    /// std library of rust).
    fn ext_max(self, other: Self) -> Self;

    /// The maximum value that can be represented by this type.
    fn ext_max_value() -> Self;
}

#[duplicate_item(T; [u8]; [u16]; [u32]; [u64]; [u128]; [usize];)]
impl ExtReal for T {
    fn ext_floor_divide(self, other: Self) -> Self {
        Integer::div_floor(&self, &other)
    }
    fn ext_min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[duplicate_item(T; [i8]; [i16]; [i32]; [i64]; [i128]; [isize];)]
impl ExtReal for T {
    fn ext_floor_divide(self, other: Self) -> Self {
        Integer::div_floor(&self, &other)
    }
    fn ext_min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[duplicate_item(T; [f32]; [f64];)]
impl ExtReal for T {
    fn ext_floor_divide(self, other: Self) -> Self {
        Float::floor(self / other)
    }
    fn ext_min(self, other: Self) -> Self {
        T::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        T::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}

#[cfg(feature = "half")]
#[duplicate_item(T; [half::f16]; [half::bf16];)]
impl ExtReal for T {
    fn ext_floor_divide(self, other: Self) -> Self {
        Float::floor(self / other)
    }
    fn ext_min(self, other: Self) -> Self {
        T::min(self, other)
    }
    fn ext_min_value() -> Self {
        Self::MIN
    }
    fn ext_max(self, other: Self) -> Self {
        T::max(self, other)
    }
    fn ext_max_value() -> Self {
        Self::MAX
    }
}
