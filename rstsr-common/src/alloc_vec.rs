extern crate alloc;
use crate::prelude_dev::*;
use core::ptr::NonNull;

/// Create an uninitialized vector with the given size.
///
/// This function depends on `aligned_alloc` feature.
/// If `aligned_alloc` is enabled, it will align at 64-bit when size of vector
/// elements is larger than 128.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
///
/// # OS system dependent
///
/// Current implementation of aligned allocation may be UB on Windows. We will disable aligned
/// allocation on platforms other than Linux and MacOS until we find a better solution.
///
/// See also <https://gitee.com/restgroup/rest_libcint/pulls/6>.
#[doc = include_str!("alloc_vec_contract.md")]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Result<Vec<T>> {
    #[cfg(all(not(target_os = "linux"), not(target_os = "macos")))]
    return unaligned_uninitialized_vec(size);

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        #[cfg(not(feature = "aligned_alloc"))]
        return unaligned_uninitialized_vec(size);
        #[cfg(feature = "aligned_alloc")]
        return aligned_uninitialized_vec::<T, 128>(size, 64);
    }
}

/// Create an unaligned uninitialized vector with the given size.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
#[doc = include_str!("alloc_vec_contract.md")]
#[allow(clippy::uninit_vec)]
pub unsafe fn unaligned_uninitialized_vec<T>(size: usize) -> Result<Vec<T>> {
    let mut v: Vec<T> = vec![];
    v.try_reserve_exact(size)?;
    // SAFETY: capacity >= `size` was reserved by `try_reserve_exact` above; per the
    // # Safety contract, the caller must initialize all elements before any read.
    unsafe { v.set_len(size) };
    return Ok(v);
}

/// Create an uninitialized vector with the given size and alignment.
///
/// - Error: `LayoutError` if the layout cannot be created.
/// - Ok(None): if the size is 0 or allocation fails.
/// - Ok(Some): pointer to the allocated memory.
///
/// <https://users.rust-lang.org/t/how-can-i-allocate-aligned-memory-in-rust/33293>
pub fn aligned_alloc(numbytes: usize, alignment: usize) -> Result<Option<NonNull<()>>> {
    if numbytes == 0 {
        return Ok(None);
    }
    let layout = alloc::alloc::Layout::from_size_align(numbytes, alignment)?;
    // SAFETY: `layout` was built from (numbytes, alignment) above; `alloc` is the
    // corresponding GlobalAlloc call, None result = allocation failure.
    let pointer = NonNull::new(unsafe { alloc::alloc::alloc(layout) }).map(|p| p.cast::<()>());
    Ok(pointer)
}

/// Create an conditionally aligned uninitialized vector with the given size.
///
/// - `N`: condition for alignment; if `N < size`, then this function will not allocate aligned
///   vector.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
#[doc = include_str!("alloc_vec_contract.md")]
#[allow(clippy::uninit_vec)]
pub unsafe fn aligned_uninitialized_vec<T, const N: usize>(size: usize, alignment: usize) -> Result<Vec<T>> {
    if size == 0 {
        return Ok(vec![]);
    } else if size < N {
        return unaligned_uninitialized_vec(size);
    } else {
        let sizeof = core::mem::size_of::<T>();
        // byte count must be overflow-checked: a wrapping `size * sizeof` would
        // allocate a (possibly tiny) buffer and then build a `Vec` claiming
        // `size` elements — out-of-bounds from the first use
        let numbytes = match size.checked_mul(sizeof) {
            Some(numbytes) => numbytes,
            None => rstsr_raise!(RuntimeError, "Allocation failed (size * size_of::<T>() overflows usize)")?,
        };
        let pointer = aligned_alloc(numbytes, alignment)?;
        if let Some(pointer) = pointer {
            // SAFETY: `pointer` comes from `aligned_alloc(size * size_of::<T>(), alignment)`
            // (this very size, above), so the Vec claims exactly the allocated extent.
            // NOTE: `alignment` (64 for `uninitialized_vec`) must also satisfy
            // `align_of::<T>()` — fine for all numeric element types used by rstsr.
            // Per the # Safety contract, the caller initializes all elements before reads.
            let mut v = Vec::from_raw_parts(pointer.as_ptr() as *mut T, size, size);
            unsafe { v.set_len(size) };
            return Ok(v);
        } else {
            rstsr_raise!(RuntimeError, "Allocation failed (probably due to out-of-memory)")?
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_aligned_alloc_overflow() {
        // Regression: `size * size_of::<T>()` used to wrap in release, so an
        // absurd element count produced a tiny allocation reinterpreted as a
        // huge `Vec` (out-of-bounds on first use). It must now be an error.
        let size = usize::MAX / 4 + 1; // * 8 (u64) overflows usize
        let r = unsafe { aligned_uninitialized_vec::<u64, 128>(size, 64) };
        assert!(r.is_err());
    }
}
