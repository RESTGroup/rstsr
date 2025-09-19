use crate::prelude_dev::*;

pub fn index_select_cpu_rayon<T, D>(
    c: &mut [MaybeUninit<T>],
    lc: &Layout<D>,
    a: &[T],
    la: &Layout<D>,
    axis: usize,
    indices: &[usize],
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    // if not in pool environment, use serial
    if pool.is_none() {
        return index_select_cpu_serial(c, lc, a, la, axis, indices);
    }

    // basic check
    let ndim = lc.ndim();
    rstsr_assert_eq!(ndim, la.ndim(), InvalidLayout, "Input and output ndim should same.")?;
    rstsr_pattern!(axis, 0..ndim, InvalidLayout, "Invalid axis that exceeds ndim.")?;
    rstsr_assert_eq!(lc.shape()[axis], indices.len(), InvalidLayout, "Invalid index length.")?;
    rstsr_pattern!(*indices.iter().max().unwrap_or(&0), 0..la.shape()[axis], InvalidLayout, "Index out of range.",)?;

    // determine how iteration should be performed
    let axis_contig_c = lc.stride()[axis] == 1;
    let size_indices = indices.len();

    if axis_contig_c {
        let lc_rest = lc.clone().dim_select(axis as isize, 0)?;
        let la_rest = la.clone().dim_select(axis as isize, 0)?;
        let layouts_rest = translate_to_col_major(&[&lc_rest, &la_rest], TensorIterOrder::K)?;
        let lc_rest = &layouts_rest[0];
        let la_rest = &layouts_rest[1];

        let axis_contig_a = la.stride()[axis] == 1;
        if axis_contig_a {
            // both axis are contiguous
            let func = |(idx_c, idx_a): (usize, usize)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut MaybeUninit<T>;
                (0..size_indices).for_each(|idx| {
                    (*c_ptr.add(idx)).write(a[idx_a + indices[idx]].clone());
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc_rest, la_rest, func);
            pool.map_or_else(task, |pool| pool.install(task))
        } else {
            let axis_stride_a = la.stride()[axis];
            let func = |(idx_c, idx_a): (usize, usize)| unsafe {
                let c_ptr = c.as_ptr().add(idx_c) as *mut MaybeUninit<T>;
                (0..size_indices).for_each(|idx| {
                    let idx_a_out = idx_a as isize + axis_stride_a * indices[idx] as isize;
                    (*c_ptr.add(idx)).write(a[idx_a_out as usize].clone());
                });
            };
            let task = || layout_col_major_dim_dispatch_par_2(lc_rest, la_rest, func);
            pool.map_or_else(task, |pool| pool.install(task))
        }
    } else {
        (0..size_indices).try_for_each(|idx| {
            let lc_selected = lc.dim_select(axis as isize, idx as isize)?;
            let la_selected = la.dim_select(axis as isize, indices[idx] as isize)?;
            assign_uninit_cpu_rayon(c, &lc_selected, a, &la_selected, pool)
        })
    }
}
