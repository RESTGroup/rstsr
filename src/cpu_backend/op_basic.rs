//! Basic math operations.
//!
//! This file assumes that layouts are pre-processed and valid.

use crate::prelude_dev::*;
use core::ops::Add;
use num::Zero;

/// Fold over the manually unrolled `xs` with `f`
///
/// # See also
///
/// This code is from <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
where
    A: Clone,
    I: Fn() -> A,
    F: Fn(A, A) -> A,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        p2 = f(p2, xs[2].clone());
        p3 = f(p3, xs[3].clone());
        p4 = f(p4, xs[4].clone());
        p5 = f(p5, xs[5].clone());
        p6 = f(p6, xs[6].clone());
        p7 = f(p7, xs[7].clone());

        xs = &xs[8..];
    }
    acc = f(acc.clone(), f(p0, p4));
    acc = f(acc.clone(), f(p1, p5));
    acc = f(acc.clone(), f(p2, p6));
    acc = f(acc.clone(), f(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for (i, x) in xs.iter().enumerate() {
        if i >= 7 {
            break;
        }
        acc = f(acc.clone(), x.clone())
    }
    acc
}

impl<T, DC, DA> OpAssignAPI<T, DC, DA> for CpuDevice
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary_layout(
        &self,
        c: &mut Storage<T, Self>,
        lc: &Layout<DC>,
        a: &Storage<T, Self>,
        la: &Layout<DA>,
    ) -> Result<()> {
        if lc.c_contig() && la.c_contig() || lc.f_contig() && la.f_contig() {
            let offset_c = lc.offset();
            let offset_a = la.offset();
            let size = lc.size();
            for i in 0..size {
                c.rawvec[offset_c + i] = a.rawvec[offset_a + i].clone();
            }
        } else {
            let order = {
                if lc.c_prefer() && la.c_prefer() {
                    TensorIterOrder::C
                } else if lc.f_prefer() && la.f_prefer() {
                    TensorIterOrder::F
                } else {
                    match TensorOrder::default() {
                        TensorOrder::C => TensorIterOrder::C,
                        TensorOrder::F => TensorIterOrder::F,
                    }
                }
            };
            let lc = translate_to_col_major_unary(lc, order)?;
            let la = translate_to_col_major_unary(la, order)?;
            let iter_c = IterLayoutColMajor::new(&lc)?;
            let iter_a = IterLayoutColMajor::new(&la)?;
            for (idx_c, idx_a) in izip!(iter_c, iter_a) {
                c.rawvec[idx_c] = a.rawvec[idx_a].clone();
            }
        }
        return Ok(());
    }
}

impl<T, D> OpAddAPI<T, D> for CpuDevice
where
    T: Add<Output = T> + Clone,
    D: DimAPI,
{
    fn ternary_add(
        &self,
        c: &mut Storage<T, CpuDevice>,
        lc: &Layout<D>,
        a: &Storage<T, CpuDevice>,
        la: &Layout<D>,
        b: &Storage<T, CpuDevice>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let layouts_full = translate_to_col_major(&[lc, la, lb], TensorIterOrder::K)?;
        let layouts_full_ref = layouts_full.iter().collect_vec();
        let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

        if size_contig >= 16 {
            let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
            let iter_b = IterLayoutColMajor::new(&layouts_contig[2])?;
            for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                for i in 0..size_contig {
                    c.rawvec[idx_c + i] = a.rawvec[idx_a + i].clone() + b.rawvec[idx_b + i].clone();
                }
            }
        } else {
            let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
            let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
            let iter_b = IterLayoutColMajor::new(&layouts_full[2])?;
            for (idx_c, idx_a, idx_b) in izip!(iter_c, iter_a, iter_b) {
                c.rawvec[idx_c] = a.rawvec[idx_a].clone() + b.rawvec[idx_b].clone();
            }
        }
        return Ok(());
    }
}

impl<T, D> OpSumAPI<T, D> for CpuDevice
where
    T: Zero + Add<Output = T> + Clone,
    D: DimAPI,
{
    fn sum(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T> {
        let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
        let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

        if size_contig >= 16 {
            let mut sum = T::zero();
            let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
            for idx_a in iter_a {
                let slc = &a.rawvec[idx_a..idx_a + size_contig];
                sum = sum + unrolled_fold(slc, || T::zero(), |acc, x| acc + x.clone());
            }
            return Ok(sum);
        } else {
            let iter_a = IterLayoutColMajor::new(&layout)?;
            let sum = iter_a.fold(T::zero(), |acc, idx| acc + a.rawvec[idx].clone());
            return Ok(sum);
        }
    }
}
