use crate::prelude_dev::*;
use num::Zero;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 128;

/// Fold over the manually unrolled `xs` with `f`
///
/// # See also
///
/// This code is from <https://github.com/rust-ndarray/ndarray/blob/master/src/numeric_util.rs>
pub fn unrolled_reduce<A, I, F>(mut xs: &[A], init: I, f: F) -> A
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

pub fn reduce_all_cpu_serial<T, D, I, F>(a: &[T], la: &Layout<D>, init: I, f: F) -> Result<T>
where
    T: Clone,
    D: DimAPI,
    I: Fn() -> T,
    F: Fn(T, T) -> T,
{
    let layout = translate_to_col_major_unary(la, TensorIterOrder::K)?;
    let (layout_contig, size_contig) = translate_to_col_major_with_contig(&[&layout]);

    if size_contig >= CONTIG_SWITCH {
        let mut acc = init();
        let iter_a = IterLayoutColMajor::new(&layout_contig[0])?;
        for idx_a in iter_a {
            let slc = &a[idx_a..idx_a + size_contig];
            let acc_inner = unrolled_reduce(slc, &init, &f);
            acc = f(acc, acc_inner);
        }
        return Ok(acc);
    } else {
        let iter_a = IterLayoutColMajor::new(&layout)?;
        let acc = iter_a.fold(init(), |acc, idx| f(acc, a[idx].clone()));
        return Ok(acc);
    }
}

#[allow(clippy::uninit_vec)]
pub fn reduce_axes_cpu_serial<T, I, F>(
    a: &[T],
    la: &Layout<IxD>,
    axes: &[isize],
    init: I,
    f: F,
) -> Result<(Vec<T>, Layout<IxD>)>
where
    T: Clone,
    I: Fn() -> T,
    F: Fn(T, T) -> T,
{
    // split the layout into axes (to be summed) and the rest
    let (layout_axes, layout_rest) = la.dim_split_axes(axes)?;

    // generate layout for result (from layout_rest)
    let layout_out = layout_for_array_copy(&layout_rest, TensorIterOrder::default())?;

    // use contiguous reduce_all only when size of contiguous part is large enough
    let (_, size_contig) = translate_to_col_major_with_contig(&[&layout_axes]);
    if size_contig >= CONTIG_SWITCH {
        // generate layouts for actual evaluation
        let layouts_swapped =
            translate_to_col_major(&[&layout_out, &layout_rest], TensorIterOrder::default())?;
        let layout_out_swapped = &layouts_swapped[0];
        let layout_rest_swapped = &layouts_swapped[1];

        // iterate both layout_rest and layout_out
        let iter_out_swapped = IterLayoutRowMajor::new(layout_out_swapped)?;
        let iter_rest_swapped = IterLayoutRowMajor::new(layout_rest_swapped)?;

        // inner layout is axes to be summed
        let mut layout_inner = layout_axes.clone();

        // prepare output
        let len_out = layout_out.size();
        let mut out = Vec::with_capacity(len_out);
        unsafe { out.set_len(len_out) };

        // actual evaluation
        izip!(iter_out_swapped, iter_rest_swapped).try_for_each(
            |(idx_out, idx_rest)| -> Result<()> {
                unsafe { layout_inner.set_offset(idx_rest) };
                let acc = reduce_all_cpu_serial(a, &layout_inner, &init, &f)?;
                out[idx_out] = acc;
                Ok(())
            },
        )?;
        return Ok((out, layout_out));
    } else {
        // iterate layout_axes
        let iter_layout_axes = IterLayoutRowMajor::new(&layout_axes)?;

        // inner layout is axes not to be summed
        let mut layout_inner = layout_rest.clone();

        // prepare output
        let len_out = layout_out.size();
        let init_val = init();
        let mut out = vec![init_val; len_out];

        // closure for adding to mutable reference
        let f_add = |a: &mut T, b: &T| *a = f(a.clone(), b.clone());

        for idx_axes in iter_layout_axes {
            unsafe { layout_inner.set_offset(idx_axes) };
            op_muta_refb_func_cpu_serial(&mut out, &layout_out, a, &layout_inner, f_add)?;
        }
        return Ok((out, layout_out));
    }
}

impl<T, D> OpSumAPI<T, D> for DeviceCpuSerial
where
    T: Zero + core::ops::Add<Output = T> + Clone,
    D: DimAPI,
{
    fn sum_all(&self, a: &Storage<T, Self>, la: &Layout<D>) -> Result<T> {
        let a = a.rawvec();
        reduce_all_cpu_serial(a, la, T::zero, |acc, x| acc + x)
    }

    fn sum(
        &self,
        a: &Storage<T, Self>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<T, Self>, Layout<IxD>)> {
        let a = a.rawvec();
        let (out, layout_out) =
            reduce_axes_cpu_serial(a, &la.to_dim()?, axes, T::zero, |acc, x| acc + x)?;
        Ok((Storage::new(out, self.clone()), layout_out))
    }
}
