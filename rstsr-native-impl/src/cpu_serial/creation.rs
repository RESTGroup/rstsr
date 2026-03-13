use num::{FromPrimitive, ToPrimitive};

use crate::prelude_dev::*;
use core::ops::*;

/* #region arange */

pub fn arange_by_partial_ord_cpu_serial<T>(start: T, end: T, step: T) -> Vec<T>
where
    T: PartialOrd + Clone + Add<Output = T>,
{
    // This is serial implementation and low performance
    let mut result = Vec::new();
    let mut current = start;
    while current < end {
        result.push(current.clone());
        current = current + step.clone();
    }
    result
}

pub fn arange_by_primitive_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    // This will use rust's internal iterator, should be much faster.
    // However, it only works for types that from/to primitives implemented.

    // We just try to convert to isize. We believe it's really rare case that usize is used in arange;
    // and anyway, fallback is always available with efficiency loss.
    let (start, end, step) = (start.to_isize()?, end.to_isize()?, step.to_isize()?);
    let n = ((end - start) as f64 / step as f64).ceil().to_usize()?;
    if step == 1 {
        (start..end).map(|x| T::from_isize(x)).collect()
    } else {
        (0..n).map(|i| T::from_isize(start + i as isize * step)).collect()
    }
}

/// Create a 1-D array of evenly spaced values within a given interval, using CPU serial
/// implementation.
///
/// Some numerical types (e.g., isize, usize) will be accelerated using internal iterators, while
/// others will use a generic implementation.
pub fn arange_cpu_serial<T>(start: T, end: T, step: T) -> Vec<T>
where
    T: PartialOrd + Clone + Add<Output = T> + 'static,
{
    use core::any::TypeId;
    use core::mem::*;

    // 1. transmute type without affecting input/output types
    #[inline]
    unsafe fn transmute_accelerated<T, U>(start: &T, end: &T, step: &T) -> Option<Vec<T>>
    where
        U: PartialOrd + Clone + Add<Output = U> + ToPrimitive + FromPrimitive,
    {
        let (start, end, step) =
            (transmute_copy::<T, U>(start), transmute_copy::<T, U>(end), transmute_copy::<T, U>(step));
        arange_by_primitive_cpu_serial(start, end, step).map(|result| transmute::<Vec<U>, Vec<T>>(result))
    }

    // 2. convenient macro in match definition
    macro_rules! expand_accelerated { ($($ty: ty),*) => {
        match TypeId::of::<T>() {
            $(id if id == TypeId::of::<$ty>() => {
                if let Some(result) = unsafe { transmute_accelerated::<T, $ty>(&start, &end, &step) } {
                    return result;
                }
            },)*
            _ => {}
        }
    }}

    // 3. types to be specialized
    expand_accelerated!(f64, f32, usize, isize, i32, i64, u32, u64);

    // 4. fallback to generic implementation
    arange_by_partial_ord_cpu_serial(start, end, step)
}

/* #endregion */
