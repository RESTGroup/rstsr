use crate::prelude_dev::*;
use num::complex::ComplexFloat;

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
        let (b_prev, b_next) = b.split_at(i + 1);
        a = a_next;
        b = b_next;
        a_prev.clone_from_slice(b_prev);
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
    let f_contig = la_inner.f_contig() && lb_inner.f_contig();

    match uplo {
        FlagUpLo::U => match (c_contig, f_contig) {
            (true, _) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_triu_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            (_, true) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_tril_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            _ => {
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
        FlagUpLo::L => match (c_contig, f_contig) {
            (true, _) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_tril_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            (_, true) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_pack_triu_c_contig(a, offset_a, b, offset_b, n);
                }
            },
            _ => {
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
        _ => rstsr_invalid!(uplo)?,
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
    let f_contig = la_inner.f_contig() && lb_inner.f_contig();

    match uplo {
        FlagUpLo::U => match (c_contig, f_contig) {
            (true, _) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_triu_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            (_, true) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_tril_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            _ => {
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
        FlagUpLo::L => match (c_contig, f_contig) {
            (true, _) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_tril_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            (_, true) => {
                for (offset_a, offset_b) in izip!(la_rest_iter, lb_rest_iter) {
                    inner_unpack_triu_c_contig(a, offset_a, b, offset_b, n, symm);
                }
            },
            _ => {
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
        _ => rstsr_invalid!(uplo)?,
    }
    Ok(())
}

/* #endregion */

/* #region trait impl */

impl<T> DeviceOpPackTriAPI<T> for DeviceCpuSerial
where
    T: Clone,
{
    fn pack_tri(
        &self,
        a: &mut Vec<T>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
    ) -> Result<()> {
        pack_tri_cpu_serial(a, la, b, lb, uplo)
    }
}

impl<T> DeviceOpUnpackTriAPI<T> for DeviceCpuSerial
where
    T: ComplexFloat,
{
    fn unpack_tri(
        &self,
        a: &mut Vec<T>,
        la: &Layout<IxD>,
        b: &Vec<T>,
        lb: &Layout<IxD>,
        uplo: FlagUpLo,
        symm: FlagSymm,
    ) -> Result<()> {
        unpack_tri_cpu_serial(a, la, b, lb, uplo, symm)
    }
}

/* #endregion */
