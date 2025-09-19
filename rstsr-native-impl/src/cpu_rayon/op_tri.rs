use crate::prelude_dev::*;
use core::slice::from_raw_parts_mut;
use core::sync::atomic::{AtomicPtr, Ordering};
use num::complex::ComplexFloat;
use rayon::prelude::*;

pub fn pack_tri_cpu_rayon<T>(
    a: &mut [MaybeUninit<T>],
    la: &Layout<IxD>,
    b: &[T],
    lb: &Layout<IxD>,
    uplo: FlagUpLo,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
{
    // we assume dimension checks have been performed, and do not check them here
    // - ndim_a = ndim_b + 1
    // - shape_a (..., n, n) and shape_b (..., n * (n + 1) / 2)
    // - rest shape are the same

    // we only parallel the rest dimensions, and inner dimensions are handled in
    // serial

    // split dimensions
    let la_split = la.dim_split_at(-1)?;
    let lb_split = lb.dim_split_at(-2)?;
    let (la_rest, la_inner) = la_split;
    let (lb_rest, lb_inner) = lb_split;

    // rest dimensions handling
    let broad_rest = translate_to_col_major(&[&la_rest, &lb_rest], TensorIterOrder::K)?;
    let la_rest = &broad_rest[0];
    let lb_rest = &broad_rest[1];
    let la_rest_iter = IterLayoutColMajor::new(la_rest)?.into_par_iter();
    let lb_rest_iter = IterLayoutColMajor::new(lb_rest)?.into_par_iter();

    // inner dimensions handling
    let n = lb_inner.shape()[0];

    // contiguous flags
    let c_contig = la_inner.c_contig() && lb_inner.c_contig();

    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());
    let len_a = a.len();

    let task = || -> Result<()> {
        match uplo {
            FlagUpLo::U => match c_contig {
                true => {
                    la_rest_iter.zip(lb_rest_iter).for_each(|(offset_a, offset_b)| {
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_pack_triu_c_contig(slice_a, offset_a, b, offset_b, n);
                    });
                },
                false => {
                    la_rest_iter.zip(lb_rest_iter).try_for_each(|(offset_a, offset_b)| -> Result<()> {
                        let mut la_inner = la_inner.to_dim::<Ix1>()?;
                        let mut lb_inner = lb_inner.to_dim::<Ix2>()?;
                        unsafe {
                            la_inner.set_offset(offset_a);
                            lb_inner.set_offset(offset_b);
                        }
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_pack_triu_general(slice_a, &la_inner, b, &lb_inner, n);
                        Ok(())
                    })?;
                },
            },
            FlagUpLo::L => match c_contig {
                true => {
                    la_rest_iter.zip(lb_rest_iter).for_each(|(offset_a, offset_b)| {
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_pack_tril_c_contig(slice_a, offset_a, b, offset_b, n);
                    });
                },
                false => {
                    la_rest_iter.zip(lb_rest_iter).try_for_each(|(offset_a, offset_b)| -> Result<()> {
                        let mut la_inner = la_inner.to_dim::<Ix1>()?;
                        let mut lb_inner = lb_inner.to_dim::<Ix2>()?;
                        unsafe {
                            la_inner.set_offset(offset_a);
                            lb_inner.set_offset(offset_b);
                        }
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_pack_tril_general(slice_a, &la_inner, b, &lb_inner, n);
                        Ok(())
                    })?;
                },
            },
        }
        Ok(())
    };

    match pool {
        Some(pool) => pool.install(task),
        None => task(),
    }
}

pub fn unpack_tri_cpu_rayon<T>(
    a: &mut [MaybeUninit<T>],
    la: &Layout<IxD>,
    b: &[T],
    lb: &Layout<IxD>,
    uplo: FlagUpLo,
    symm: FlagSymm,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: ComplexFloat + Send + Sync,
{
    // we assume dimension checks have been performed, and do not check them here
    // - ndim_a + 1 = ndim_b
    // - shape_a (..., n * (n + 1) / 2) and shape_b (..., n, n)
    // - rest shape are the same

    // we only parallel the rest dimensions, and inner dimensions are handled in
    // serial

    // split dimensions
    let la_split = la.dim_split_at(-2)?;
    let lb_split = lb.dim_split_at(-1)?;
    let (la_rest, la_inner) = la_split;
    let (lb_rest, lb_inner) = lb_split;

    // rest dimensions handling
    let broad_rest = translate_to_col_major(&[&la_rest, &lb_rest], TensorIterOrder::K)?;
    let la_rest = &broad_rest[0];
    let lb_rest = &broad_rest[1];
    let la_rest_iter = IterLayoutColMajor::new(la_rest)?.into_par_iter();
    let lb_rest_iter = IterLayoutColMajor::new(lb_rest)?.into_par_iter();

    // inner dimensions handling
    let n = la_inner.shape()[0];

    // contiguous flags
    let c_contig = la_inner.c_contig() && lb_inner.c_contig();

    // pass mutable reference in parallel region
    let thr_a = AtomicPtr::new(a.as_mut_ptr());
    let len_a = a.len();

    let task = || -> Result<()> {
        match uplo {
            FlagUpLo::U => match c_contig {
                true => {
                    la_rest_iter.zip(lb_rest_iter).for_each(|(offset_a, offset_b)| {
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_unpack_triu_c_contig(slice_a, offset_a, b, offset_b, n, symm);
                    });
                },
                false => {
                    la_rest_iter.zip(lb_rest_iter).try_for_each(|(offset_a, offset_b)| -> Result<()> {
                        let mut la_inner = la_inner.to_dim::<Ix2>()?;
                        let mut lb_inner = lb_inner.to_dim::<Ix1>()?;
                        unsafe {
                            la_inner.set_offset(offset_a);
                            lb_inner.set_offset(offset_b);
                        }
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_unpack_triu_general(slice_a, &la_inner, b, &lb_inner, n, symm);
                        Ok(())
                    })?;
                },
            },
            FlagUpLo::L => match c_contig {
                true => {
                    la_rest_iter.zip(lb_rest_iter).for_each(|(offset_a, offset_b)| {
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_unpack_tril_c_contig(slice_a, offset_a, b, offset_b, n, symm);
                    });
                },
                false => {
                    la_rest_iter.zip(lb_rest_iter).try_for_each(|(offset_a, offset_b)| -> Result<()> {
                        let mut la_inner = la_inner.to_dim::<Ix2>()?;
                        let mut lb_inner = lb_inner.to_dim::<Ix1>()?;
                        unsafe {
                            la_inner.set_offset(offset_a);
                            lb_inner.set_offset(offset_b);
                        }
                        let ptr_a = thr_a.load(Ordering::Relaxed);
                        let slice_a = unsafe { from_raw_parts_mut(ptr_a, len_a) };
                        inner_unpack_tril_general(slice_a, &la_inner, b, &lb_inner, n, symm);
                        Ok(())
                    })?;
                },
            },
        }
        Ok(())
    };

    match pool {
        Some(pool) => pool.install(task),
        None => task(),
    }
}
