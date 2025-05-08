use crate::prelude_dev::*;
use num::complex::ComplexFloat;
use num::Num;

/* #region pack tri */

#[inline]
pub fn inner_pack_tril_c_contig<T>(a: &mut [T], offset_a: usize, b: &[T], offset_b: usize, n: usize)
where
    T: Clone,
{
    let mut a = &mut a[offset_a..];
    let mut b = &b[offset_b..];
    for i in 0..n {
        let (a_prev, a_next) = a.split_at_mut(i + 1);
        let (b_prev, b_next) = b.split_at(n);
        a = a_next;
        b = b_next;
        a_prev.iter_mut().zip(b_prev.iter()).for_each(|(ai, bi)| *ai = bi.clone());
    }
}

#[inline]
pub fn inner_pack_tril_general<T>(
    a: &mut [T],
    la: &Layout<Ix1>,
    b: &[T],
    lb: &Layout<Ix2>,
    n: usize,
) where
    T: Clone,
{
    let mut idx_a = 0;
    for i in 0..n {
        for j in 0..=i {
            let loc_b = unsafe { lb.index_uncheck(&[i, j]) } as usize;
            let loc_a = unsafe { la.index_uncheck(&[idx_a]) } as usize;
            a[loc_a] = b[loc_b].clone();
            idx_a += 1;
        }
    }
}

#[inline]
pub fn inner_pack_triu_c_contig<T>(a: &mut [T], offset_a: usize, b: &[T], offset_b: usize, n: usize)
where
    T: Clone,
{
    let a = &mut a[offset_a..];
    let b = &b[offset_b..];
    let mut idx_a = 0;
    for i in 0..n {
        for j in i..n {
            a[idx_a] = b[i * n + j].clone();
            idx_a += 1;
        }
    }
}

#[inline]
pub fn inner_pack_triu_general<T>(
    a: &mut [T],
    la: &Layout<Ix1>,
    b: &[T],
    lb: &Layout<Ix2>,
    n: usize,
) where
    T: Clone,
{
    let mut idx_a = 0;
    for i in 0..n {
        for j in i..n {
            let loc_b = unsafe { lb.index_uncheck(&[i, j]) } as usize;
            let loc_a = unsafe { la.index_uncheck(&[idx_a]) } as usize;
            a[loc_a] = b[loc_b].clone();
            idx_a += 1;
        }
    }
}

pub fn pack_tri_cpu_serial<T>(
    a: &mut [T],
    la: &Layout<IxD>,
    b: &[T],
    lb: &Layout<IxD>,
    uplo: FlagUpLo,
) -> Result<()>
where
    T: Clone,
{
    // we assume dimension checks have been performed, and do not check them here
    // - ndim_a = ndim_b + 1
    // - shape_a (..., n, n) and shape_b (..., n * (n + 1) / 2)
    // - rest shape are the same

    // split dimensions
    let la_split = la.dim_split_at(-1)?;
    let lb_split = lb.dim_split_at(-2)?;
    let (la_rest, la_inner) = la_split;
    let (lb_rest, lb_inner) = lb_split;

    // rest dimensions handling
    let broad_rest = translate_to_col_major(&[&la_rest, &lb_rest], TensorIterOrder::K)?;
    let la_rest = &broad_rest[0];
    let lb_rest = &broad_rest[1];
    let la_rest_iter = IterLayoutColMajor::new(la_rest)?;
    let lb_rest_iter = IterLayoutColMajor::new(lb_rest)?;

    // inner dimensions handling
    let n = lb_inner.shape()[0];

    // contiguous flags
    let c_contig = la_inner.c_contig() && lb_inner.c_contig();

    match uplo {
        FlagUpLo::U => match c_contig {
            true => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_triu_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            false => {
                let mut la_inner = la_inner.to_dim::<Ix1>()?;
                let mut lb_inner = lb_inner.to_dim::<Ix2>()?;
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    unsafe {
                        la_inner.set_offset(offset_a);
                        lb_inner.set_offset(offset_b);
                    }
                    inner_pack_triu_general(a, &la_inner, b, &lb_inner, n);
                }
            },
        },
        FlagUpLo::L => match c_contig {
            true => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_tril_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            false => {
                let mut la_inner = la_inner.to_dim::<Ix1>()?;
                let mut lb_inner = lb_inner.to_dim::<Ix2>()?;
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    unsafe {
                        la_inner.set_offset(offset_a);
                        lb_inner.set_offset(offset_b);
                    }
                    inner_pack_tril_general(a, &la_inner, b, &lb_inner, n);
                }
            },
        },
    }
    Ok(())
}

/* #endregion */

/* #region unpack tri */

#[inline]
pub fn inner_unpack_tril_c_contig<T>(
    a: &mut [T],
    offset_a: usize,
    b: &[T],
    offset_b: usize,
    n: usize,
    symm: FlagSymm,
) where
    T: ComplexFloat,
{
    let a = &mut a[offset_a..];
    let b = &b[offset_b..];
    let mut idx_b = 0;
    match symm {
        FlagSymm::Sy => {
            for i in 0..n {
                for j in 0..=i {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = b[idx_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::He => {
            for i in 0..n {
                for j in 0..=i {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = b[idx_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ay => {
            for i in 0..n {
                for j in 0..i {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = -b[idx_b];
                    idx_b += 1;
                }
                a[i * n + i] = T::zero();
                idx_b += 1;
            }
        },
        FlagSymm::Ah => {
            for i in 0..n {
                for j in 0..i {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = -b[idx_b].conj();
                    idx_b += 1;
                }
                a[i * n + i] = T::zero();
                idx_b += 1;
            }
        },
        FlagSymm::N => {
            for i in 0..n {
                for j in 0..=i {
                    a[i * n + j] = b[idx_b];
                    idx_b += 1;
                }
            }
        },
    }
}

#[inline]
pub fn inner_unpack_tril_general<T>(
    a: &mut [T],
    la: &Layout<Ix2>,
    b: &[T],
    lb: &Layout<Ix1>,
    n: usize,
    symm: FlagSymm,
) where
    T: ComplexFloat,
{
    let mut idx_b = 0;
    match symm {
        FlagSymm::Sy => {
            for i in 0..n {
                for j in 0..=i {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = b[loc_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::He => {
            for i in 0..n {
                for j in 0..=i {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = b[loc_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ay => {
            for i in 0..n {
                for j in 0..i {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = -b[loc_b];
                    idx_b += 1;
                }
                let loc_a_ii = unsafe { la.index_uncheck(&[i, i]) } as usize;
                a[loc_a_ii] = T::zero();
                idx_b += 1;
            }
        },
        FlagSymm::Ah => {
            for i in 0..n {
                for j in 0..i {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = -b[loc_b].conj();
                    idx_b += 1;
                }
                let loc_a_ii = unsafe { la.index_uncheck(&[i, i]) } as usize;
                a[loc_a_ii] = T::zero();
                idx_b += 1;
            }
        },
        FlagSymm::N => {
            for i in 0..n {
                for j in 0..=i {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    idx_b += 1;
                }
            }
        },
    }
}

#[inline]
pub fn inner_unpack_triu_c_contig<T>(
    a: &mut [T],
    offset_a: usize,
    b: &[T],
    offset_b: usize,
    n: usize,
    symm: FlagSymm,
) where
    T: ComplexFloat,
{
    let a = &mut a[offset_a..];
    let b = &b[offset_b..];
    let mut idx_b = 0;
    match symm {
        FlagSymm::Sy => {
            for i in 0..n {
                for j in i..n {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = b[idx_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::He => {
            for i in 0..n {
                for j in i..n {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = b[idx_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ay => {
            for i in 0..n {
                a[i * n + i] = T::zero();
                idx_b += 1;
                for j in (i + 1)..n {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = -b[idx_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ah => {
            for i in 0..n {
                a[i * n + i] = T::zero();
                idx_b += 1;
                for j in (i + 1)..n {
                    a[i * n + j] = b[idx_b];
                    a[j * n + i] = -b[idx_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::N => {
            for i in 0..n {
                for j in i..n {
                    a[i * n + j] = b[idx_b];
                    idx_b += 1;
                }
            }
        },
    }
}

#[inline]
pub fn inner_unpack_triu_general<T>(
    a: &mut [T],
    la: &Layout<Ix2>,
    b: &[T],
    lb: &Layout<Ix1>,
    n: usize,
    symm: FlagSymm,
) where
    T: ComplexFloat,
{
    let mut idx_b = 0;
    match symm {
        FlagSymm::Sy => {
            for i in 0..n {
                for j in i..n {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = b[loc_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::He => {
            for i in 0..n {
                for j in i..n {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = b[loc_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ay => {
            for i in 0..n {
                let loc_a_ii = unsafe { la.index_uncheck(&[i, i]) } as usize;
                a[loc_a_ii] = T::zero();
                idx_b += 1;
                for j in (i + 1)..n {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = -b[loc_b];
                    idx_b += 1;
                }
            }
        },
        FlagSymm::Ah => {
            for i in 0..n {
                let loc_a_ii = unsafe { la.index_uncheck(&[i, i]) } as usize;
                a[loc_a_ii] = T::zero();
                idx_b += 1;
                for j in (i + 1)..n {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    let loc_a_ji = unsafe { la.index_uncheck(&[j, i]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    a[loc_a_ji] = -b[loc_b].conj();
                    idx_b += 1;
                }
            }
        },
        FlagSymm::N => {
            for i in 0..n {
                for j in i..n {
                    let loc_b = unsafe { lb.index_uncheck(&[idx_b]) } as usize;
                    let loc_a_ij = unsafe { la.index_uncheck(&[i, j]) } as usize;
                    a[loc_a_ij] = b[loc_b];
                    idx_b += 1;
                }
            }
        },
    }
}

pub fn unpack_tri_cpu_serial<T>(
    a: &mut [T],
    la: &Layout<IxD>,
    b: &[T],
    lb: &Layout<IxD>,
    uplo: FlagUpLo,
    symm: FlagSymm,
) -> Result<()>
where
    T: ComplexFloat,
{
    // we assume dimension checks have been performed, and do not check them here
    // - ndim_a + 1 = ndim_b
    // - shape_a (..., n * (n + 1) / 2) and shape_b (..., n, n)
    // - rest shape are the same

    // split dimensions
    let la_split = la.dim_split_at(-2)?;
    let lb_split = lb.dim_split_at(-1)?;
    let (la_rest, la_inner) = la_split;
    let (lb_rest, lb_inner) = lb_split;

    // rest dimensions handling
    let broad_rest = translate_to_col_major(&[&la_rest, &lb_rest], TensorIterOrder::K)?;
    let la_rest = &broad_rest[0];
    let lb_rest = &broad_rest[1];
    let la_rest_iter = IterLayoutColMajor::new(la_rest)?;
    let lb_rest_iter = IterLayoutColMajor::new(lb_rest)?;

    // inner dimensions handling
    let n = la_inner.shape()[0];

    // contiguous flags
    let c_contig = la_inner.c_contig() && lb_inner.c_contig();

    match uplo {
        FlagUpLo::U => match c_contig {
            true => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_triu_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            false => {
                let mut la_inner = la_inner.to_dim::<Ix2>()?;
                let mut lb_inner = lb_inner.to_dim::<Ix1>()?;
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    unsafe {
                        la_inner.set_offset(offset_a);
                        lb_inner.set_offset(offset_b);
                    }
                    inner_unpack_triu_general(a, &la_inner, b, &lb_inner, n, symm);
                }
            },
        },
        FlagUpLo::L => match c_contig {
            true => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_tril_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            false => {
                let mut la_inner = la_inner.to_dim::<Ix2>()?;
                let mut lb_inner = lb_inner.to_dim::<Ix1>()?;
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    unsafe {
                        la_inner.set_offset(offset_a);
                        lb_inner.set_offset(offset_b);
                    }
                    inner_unpack_tril_general(a, &la_inner, b, &lb_inner, n, symm);
                }
            },
        },
    }
    Ok(())
}

/* #endregion */

/* #region tril */

pub fn tril_cpu_serial<T, D>(raw: &mut [T], layout: &Layout<D>, k: isize) -> Result<()>
where
    T: Num + Clone,
    D: DimAPI,
{
    let (la_rest, la_ix2) = layout.dim_split_at(-2)?;
    let mut la_ix2 = la_ix2.into_dim::<Ix2>()?;
    for offset in IterLayoutColMajor::new(&la_rest)? {
        unsafe { la_ix2.set_offset(offset) };
        tril_ix2_cpu_serial(raw, &la_ix2, k)?;
    }
    Ok(())
}

pub fn tril_ix2_cpu_serial<T>(raw: &mut [T], layout: &Layout<Ix2>, k: isize) -> Result<()>
where
    T: Num + Clone,
{
    let [nrow, ncol] = *layout.shape();
    for i in 0..nrow {
        let j_start = (i as isize + k + 1).max(0) as usize;
        for j in j_start..ncol {
            unsafe {
                raw[layout.index_uncheck(&[i, j]) as usize] = T::zero();
            }
        }
    }
    Ok(())
}

/* #endregion */

/* #region triu */

pub fn triu_cpu_serial<T, D>(raw: &mut [T], layout: &Layout<D>, k: isize) -> Result<()>
where
    T: Num + Clone,
    D: DimAPI,
{
    let (la_rest, la_ix2) = layout.dim_split_at(-2)?;
    let mut la_ix2 = la_ix2.into_dim::<Ix2>()?;
    for offset in IterLayoutColMajor::new(&la_rest)? {
        unsafe { la_ix2.set_offset(offset) };
        triu_ix2_cpu_serial(raw, &la_ix2, k)?;
    }
    Ok(())
}

pub fn triu_ix2_cpu_serial<T>(raw: &mut [T], layout: &Layout<Ix2>, k: isize) -> Result<()>
where
    T: Num + Clone,
{
    let [nrow, _] = *layout.shape();
    for i in 0..nrow {
        let j_end = (i as isize + k).max(0) as usize;
        for j in 0..j_end {
            unsafe {
                raw[layout.index_uncheck(&[i, j]) as usize] = T::zero();
            }
        }
    }
    Ok(())
}

/* #endregion */
