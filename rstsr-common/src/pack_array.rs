//! Cast `Vec<T>` to/from `Vec<[T; N]>` (or its reference) without copying.

use crate::prelude_dev::*;
use std::mem::ManuallyDrop;

pub trait PackArrayAPI<const N: usize> {
    type Output;

    fn pack_array_f(self) -> Result<Self::Output>;
    fn pack_array(self) -> Self::Output
    where
        Self: Sized,
    {
        self.pack_array_f().unwrap()
    }
}

pub trait UnpackArrayAPI {
    type Output;
    fn unpack_array(self) -> Self::Output;
}

pub fn pack_array<const N: usize, V>(v: V) -> V::Output
where
    V: PackArrayAPI<N>,
{
    v.pack_array()
}

pub fn unpack_array<V>(v: V) -> V::Output
where
    V: UnpackArrayAPI,
{
    v.unpack_array()
}

/* #region impl of Vec<T> */

impl<T, const N: usize> PackArrayAPI<N> for Vec<T> {
    type Output = Vec<[T; N]>;

    fn pack_array_f(self) -> Result<Self::Output> {
        let len = self.len();
        rstsr_assert!(
            len % N == 0,
            InvalidValue,
            "Length of Vec<T> {len} must be a multiple to cast into Vec<[T; {N}]>"
        )?;
        let vec = ManuallyDrop::new(self);
        let arr = unsafe { Vec::from_raw_parts(vec.as_ptr() as *mut [T; N], len / N, len / N) };
        Ok(arr)
    }
}

impl<T, const N: usize> UnpackArrayAPI for Vec<[T; N]> {
    type Output = Vec<T>;

    fn unpack_array(self) -> Self::Output {
        let len = self.len();
        let arr = ManuallyDrop::new(self);
        unsafe { Vec::from_raw_parts(arr.as_ptr() as *mut T, len * N, len * N) }
    }
}

/* #endregion */

/* #region impl of &[T] */

impl<'l, T, const N: usize> PackArrayAPI<N> for &'l [T] {
    type Output = &'l [[T; N]];

    fn pack_array_f(self) -> Result<Self::Output> {
        let len = self.len();
        rstsr_assert!(
            len % N == 0,
            InvalidValue,
            "Length of &[T] {len} must be a multiple to cast into Vec<[T; {N}]>"
        )?;
        let arr = unsafe { core::slice::from_raw_parts(self.as_ptr() as *mut [T; N], len / N) };
        Ok(arr)
    }
}

impl<'l, T, const N: usize> UnpackArrayAPI for &'l [[T; N]] {
    type Output = &'l [T];

    fn unpack_array(self) -> Self::Output {
        let len = self.len();
        unsafe { core::slice::from_raw_parts(self.as_ptr() as *mut T, len * N) }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        let v1 = vec![1, 2, 3, 4, 5, 6];
        let ptr_v1 = v1.as_ptr();

        let a1: Vec<[i32; 2]> = v1.pack_array();
        let ptr_a1 = a1.as_ptr();
        assert_eq!(a1, vec![[1, 2], [3, 4], [5, 6]]);
        assert_eq!(ptr_v1, ptr_a1 as *const i32);

        let v2: Vec<i32> = a1.unpack_array();
        let ptr_v2 = v2.as_ptr();
        assert_eq!(v2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(ptr_v1, ptr_v2);
    }
}
