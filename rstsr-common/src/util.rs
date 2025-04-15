use crate::prelude_dev::*;
use core::mem::{transmute, MaybeUninit};

/* #region uninitialized vector */

/// Create an uninitialized vector with the given size.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(size);
    unsafe { v.set_len(size) };
    return unsafe { transmute::<Vec<MaybeUninit<T>>, Vec<T>>(v) };
}

/* #endregion */

/* #region trait for split_at */

pub trait IterSplitAtAPI: Sized {
    // Function that split the iterator at the given index.
    // This is used for parallel iterator.
    fn split_at(self, index: usize) -> (Self, Self);
}

/* #endregion */
