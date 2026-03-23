use crate::prelude_dev::*;
use core::ops::*;
use num::{FromPrimitive, ToPrimitive};

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

pub fn arange_by_primitive_isize_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    // This will use rust's internal iterator, should be much faster.
    // However, it only works for types that from/to primitives implemented.
    // Also, it only works for integer types because of the precision issue.

    // We just try to convert to isize. We believe it's really rare case that usize is used in arange;
    // and anyway, fallback is always available with efficiency loss.
    let (start_, end_, step_) = (start.to_isize()?, end.to_isize()?, step.to_isize()?);
    let n = ((end_ - start_) as f64 / step_ as f64).ceil().to_isize()?;

    // The unwrap here is probably safe, since start/end/nstep are all checked.
    // `Some(map(option.unwrap).collect())` is much faster than `map(option).collect()` in this case, so
    // decided unwrap here.
    let mut result: Vec<T> = (0..n).map(|i| T::from_isize(start_ + i * step_).unwrap()).collect();

    // the interval may be open on the right, so we need to pop the last element if it's out of range.
    let last_val = result.last().cloned();
    if (step_ > 0 && last_val.as_ref().is_some_and(|x| *x >= end))
        || (step_ < 0 && last_val.as_ref().is_some_and(|x| *x <= end))
    {
        result.pop();
    }
    Some(result)
}

pub fn arange_by_primitive_f64_cpu_serial<T>(start: T, end: T, step: T) -> Option<Vec<T>>
where
    T: PartialOrd + Clone + Add<Output = T> + ToPrimitive + FromPrimitive,
{
    let (start_, end_, step_) = (start.to_f64()?, end.to_f64()?, step.to_f64()?);
    let n = ((end_ - start_) / step_).ceil().to_usize()?;
    let mut result: Vec<T> = (0..n).map(|i| T::from_f64(start_ + i as f64 * step_).unwrap()).collect();

    // the interval may be open on the right, so we need to pop the last element if it's out of range.
    let last_val = result.last().cloned();
    if (step_ > 0.0 && last_val.as_ref().is_some_and(|x| *x >= end))
        || (step_ < 0.0 && last_val.as_ref().is_some_and(|x| *x <= end))
    {
        result.pop();
    }
    Some(result)
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
    unsafe fn transmute_accelerated<T, U>(start: &T, end: &T, step: &T, int_mode: bool) -> Option<Vec<T>>
    where
        U: PartialOrd + Clone + Add<Output = U> + ToPrimitive + FromPrimitive,
    {
        let (start, end, step) =
            (transmute_copy::<T, U>(start), transmute_copy::<T, U>(end), transmute_copy::<T, U>(step));
        let v = match int_mode {
            true => arange_by_primitive_isize_cpu_serial(start, end, step),
            false => arange_by_primitive_f64_cpu_serial(start, end, step),
        };
        v.map(|result| transmute::<Vec<U>, Vec<T>>(result))
    }

    // 2. convenient macro in match definition
    macro_rules! expand_accelerated { ($mode: ident, $($ty: ty),*) => {
        match TypeId::of::<T>() {
            $(id if id == TypeId::of::<$ty>() => unsafe { transmute_accelerated::<T, $ty>(&start, &end, &step, $mode) },)*
            _ => None
        }
    }}

    // 3. execute accelerated implementation if possible
    expand_accelerated!(false, f64, f32).unwrap_or_else(|| {
        expand_accelerated!(true, usize, isize, i32, i64, u32, u64)
            .unwrap_or_else(|| arange_by_partial_ord_cpu_serial(start, end, step))
    })
}

/* #endregion */
