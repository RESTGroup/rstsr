//! Layout rearrangement.
//!
//! Purposes of rearrangement of layouts:
//! - Faster iteration to inplace-modify storage, and binary/ternary operations.
//! - Split layout to multiple layouts.

use crate::prelude_dev::*;

// type alias for this file
type Order = TensorIterOrder;

/* #region translate tensor order to col-major with TensorIterType */

/// This function will return a f-prefer layout that make minimal memory
/// accessing efforts (pointers will not frequently back-and-forth).
///
/// Note that this function should only be used for iteration.
///
/// # Parameter `keep_shape`
///
/// Keep size of output layout when input layout is boardcasted.
/// This option should be false if [`TensorIterOrder::K`] and true if
/// [`TensorIterOrder::G`].
///
/// For example of layout shape `[5, 1, 2, 1, 3, 6]` and stride `[1000, 10, 10,
/// 40, 0, 100]`,
/// - false: shape `[2, 6, 5, 1, 1, 1]` and stride `[10, 100, 1000, 0, 0, 0]`;
///   meaning that broadcasted shapes are eliminated and moved to last axes.
/// - true: shape `[3, 1, 1, 2, 6, 5]` and stride `[0, 10, 40, 10, 100, 1000]`;
///   meaning that broadcasted shapes are iterated with most priority.
///
/// # Returns
///
/// - `layout`: The output layout of greedy iteration.
/// - `index`: Transpose index from input layout to output layout.
pub fn greedy_layout<D>(layout: &Layout<D>, keep_shape: bool) -> (Layout<D>, Vec<isize>)
where
    D: DimDevAPI,
{
    let mut layout = layout.clone();

    // if no elements in layout, return itself
    if layout.size() == 0 {
        return (layout.clone(), (0..layout.ndim() as isize).collect_vec());
    }

    // revert negative strides if keep_shape is not required
    if keep_shape {
        for n in 0..layout.ndim() {
            if layout.stride()[n] < 0 {
                // should not panic here
                layout = layout.dim_narrow(n as isize, slice!(None, None, -1)).unwrap();
            }
        }
    }

    let shape_old = layout.shape.as_ref();
    let stride_old = layout.stride.as_ref();

    let mut index = (0..layout.ndim() as isize).collect_vec();
    if keep_shape {
        // sort shape and strides if keep shape
        // - (shape = 1 / stride = 0) the smallest (pointer not moving for these cases)
        // - if (shape = 1 / stride = 0, broadcastable axes) preserve order
        // - (larger shape first) if not broadcastable axes, then compare stride size
        //   (smaller stride first)
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1 as usize];
            let d2 = shape_old[i2 as usize];
            let t1 = stride_old[i1 as usize];
            let t2 = stride_old[i2 as usize];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => i1.cmp(&i2),
                (true, false) => core::cmp::Ordering::Less,
                (false, true) => core::cmp::Ordering::Greater,
                (false, false) => t1.abs().cmp(&t2.abs()),
            }
        });
    } else {
        // sort shape and strides if not keep shape
        // everything is similar, though broadcastable axes should be moved to last
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1 as usize];
            let d2 = shape_old[i2 as usize];
            let t1 = stride_old[i1 as usize];
            let t2 = stride_old[i2 as usize];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => i1.cmp(&i2),
                (true, false) => core::cmp::Ordering::Greater,
                (false, true) => core::cmp::Ordering::Less,
                (false, false) => t1.abs().cmp(&t2.abs()),
            }
        });
    }

    let mut layout = layout.transpose(&index).unwrap();

    // for case of not keep shape, dimension of broadcastable axes will be set to 1,
    // strides will be set to 0.
    if !keep_shape {
        let mut shape = layout.shape().clone();
        let mut stride = layout.stride().clone();
        shape.as_mut().iter_mut().zip(stride.as_mut().iter_mut()).for_each(|(d, t)| {
            if *d == 1 || *t == 0 {
                *d = 1;
                *t = 0;
            }
        });
        layout = unsafe { Layout::new_unchecked(shape, stride, layout.offset()) };
    }

    return (layout, index);
}

/// Reversed permutation indices.
pub fn reversed_permute(indices: &[isize]) -> Vec<isize> {
    let mut new_indices = vec![0; indices.len()];
    for (idx, &i) in indices.iter().enumerate() {
        new_indices[i as usize] = idx as isize;
    }
    return new_indices;
}

/// Return a layout that is suitable for array copy.
pub fn layout_for_array_copy<D>(layout: &Layout<D>, order: TensorIterOrder) -> Result<Layout<D>>
where
    D: DimDevAPI,
{
    let layout = match order {
        Order::C => layout.shape().c(),
        Order::F => layout.shape().f(),
        Order::A => {
            if layout.c_contig() {
                layout.shape().c()
            } else if layout.f_contig() {
                layout.shape().f()
            } else {
                match TensorOrder::default() {
                    RowMajor => layout.shape().c(),
                    ColMajor => layout.shape().f(),
                }
            }
        },
        Order::K => {
            let (greedy, indices) = greedy_layout(layout, true);
            let layout = greedy.shape().f();
            layout.transpose(&reversed_permute(&indices))?
        },
        _ => rstsr_invalid!(order, "Iter order for copy only accepts CFAK.")?,
    };
    return Ok(layout);
}

/// Translate one layout to column-major iteration.
///
/// For how parameter `it_type` works, we refer to definition of
/// [`TensorIterOrder`].
///
/// - C: reverse axes
/// - F: preserve axes
/// - A: B if contiguous, C if c-prefer, F if f-prefer; otherwise default
/// - K: greedy layout, keep shape
/// - G: greedy layout, eliminate broadcastable dimensions
/// - B: sequential memory; valid option if `size = bound_max - bound_min`,
///   otherwise raise err
pub fn translate_to_col_major_unary<D>(
    layout: &Layout<D>,
    order: TensorIterOrder,
) -> Result<Layout<D>>
where
    D: DimDevAPI,
{
    let fn_c = |l: &Layout<D>| Ok(l.reverse_axes());
    let fn_f = |l: &Layout<D>| Ok(l.clone());
    let fn_b = |l: &Layout<D>| {
        let (bounds_min, bounds_max) = l.bounds_index()?;
        rstsr_assert_eq!(
            bounds_max - bounds_min,
            l.size(),
            InvalidLayout,
            "Data in this layout could not be represented as sequential memory."
        )?;
        let mut shape = l.new_shape();
        let mut stride = l.new_stride();
        shape[0] = l.size();
        stride[0] = 1;
        for i in 1..l.ndim() {
            shape[i] = 1;
            stride[i] = l.size() as isize;
        }
        Ok(unsafe { Layout::new_unchecked(shape, stride, l.offset()) })
    };
    match order {
        Order::C => fn_c(layout),
        Order::F => fn_f(layout),
        Order::A => {
            let c_contig = layout.c_contig();
            let f_contig = layout.f_contig();
            if c_contig || f_contig {
                fn_b(layout)
            } else {
                let c_prefer = layout.c_prefer();
                let f_prefer = layout.f_prefer();
                match (c_prefer, f_prefer) {
                    (true, false) => fn_c(layout),
                    (false, true) => fn_f(layout),
                    (_, _) => match FlagOrder::default() {
                        RowMajor => fn_c(layout),
                        ColMajor => fn_f(layout),
                    },
                }
            }
        },
        Order::K => Ok(greedy_layout(layout, true).0),
        Order::G => Ok(greedy_layout(layout, false).0),
        Order::B => fn_b(layout),
    }
}

/// Translate multiple layouts to column-major iteration.
///
/// This function requires all layouts have the same shape.
///
/// For how parameter `it_type` works, we refer to definition of
/// [`TensorIterOrder`].
///
/// - C: reverse axes
/// - F: preserve axes
/// - A: B if contiguous, C if c-prefer, F if f-prefer; otherwise default
/// - K: greedy layout for the one which have the largest non-broadcast-size,
///   otherwise left-most layout (usually for mutable-assign/inplace-op)
/// - G: invalid option here
/// - B: sequential memory; valid option if `size = bound_max - bound_min`,
///   otherwise raise err
///
/// This operation will not flip any strides.
pub fn translate_to_col_major<D>(
    layouts: &[&Layout<D>],
    order: TensorIterOrder,
) -> Result<Vec<Layout<D>>>
where
    D: DimAPI,
{
    if layouts.is_empty() {
        return Ok(vec![]);
    }

    // this function will map all layouts to column-major iteration by a single
    // iter-order.
    let fn_single = |ls: &[&Layout<D>], order| {
        ls.iter().map(|l| translate_to_col_major_unary(l, order)).collect()
    };

    // make sure all layouts have the same shape
    let is_same_shape = layouts.windows(2).all(|w| w[0].shape() == w[1].shape());
    rstsr_assert!(
        is_same_shape,
        InvalidLayout,
        "All shape of layout in this function must be the same."
    )?;

    match order {
        Order::C | Order::F | Order::B => fn_single(layouts, order),
        Order::A => {
            let c_contig = layouts.iter().all(|&l| l.c_contig());
            let f_contig = layouts.iter().all(|&l| l.f_contig());
            if c_contig || f_contig {
                fn_single(layouts, Order::B)
            } else {
                let c_prefer = layouts.iter().all(|&l| l.c_contig());
                let f_prefer = layouts.iter().all(|&l| l.f_contig());
                match (c_prefer, f_prefer) {
                    (true, false) => fn_single(layouts, Order::C),
                    (false, true) => fn_single(layouts, Order::F),
                    (_, _) => match FlagOrder::default() {
                        RowMajor => fn_single(layouts, Order::C),
                        ColMajor => fn_single(layouts, Order::F),
                    },
                }
            }
        },
        Order::K => {
            // find the layout with the largest non-broadcast-size
            let size_iter = layouts.iter().map(|l| l.size_non_broadcast()).collect_vec();
            let idx_layout = if size_iter.iter().max() == size_iter.iter().min() {
                0
            } else {
                size_iter.into_iter().enumerate().max_by_key(|(_, v)| *v).unwrap_or((0, 0)).0
            };
            // make same permutation for all layouts
            let (_, permute_index) = greedy_layout(layouts[idx_layout], true);
            layouts.iter().map(|l| l.transpose(&permute_index)).collect()
        },
        Order::G => rstsr_invalid!(order, "This option is not valid for multiple layouts")?,
    }
}

/// This function will return minimal dimension layout, that the first axis is
/// f-contiguous.
///
/// For example, if shape [2, 4, 6, 8, 10] is contiguous in f-order for the
/// first three axes, then it will return shape [48, 8, 10], and the contiguous
/// size 48.
///
/// # Notes
///
/// - Should be used after [`translate_to_col_major`].
/// - Accepts multiple layouts to be compared.
/// - Due to that final dimension is not known to compiler, this function will
///   return dynamic layout.
pub fn translate_to_col_major_with_contig<D>(layouts: &[&Layout<D>]) -> (Vec<Layout<IxD>>, usize)
where
    D: DimAPI,
{
    if layouts.is_empty() {
        return (vec![], 0);
    }

    let dims_f_contig = layouts.iter().map(|l| l.ndim_of_f_contig()).collect_vec();
    let ndim_f_contig = *dims_f_contig.iter().min().unwrap();
    // following is the worst case: no axes are contiguous in f-order
    if ndim_f_contig == 0 {
        return (layouts.iter().map(|&l| l.clone().into_dim::<IxD>().unwrap()).collect(), 0);
    } else {
        let size_contig = layouts[0].shape().as_ref()[0..ndim_f_contig].iter().product::<usize>();
        let result = layouts
            .iter()
            .map(|l| {
                let shape = l.shape().as_ref()[ndim_f_contig..].iter().cloned().collect_vec();
                let stride = l.stride().as_ref()[ndim_f_contig..].iter().cloned().collect_vec();
                unsafe { Layout::new_unchecked(shape, stride, l.offset()) }
            })
            .collect_vec();
        return (result, size_contig);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_greedy_layout() {
        unsafe {
            // c-contiguous layout
            let layout = [2, 3, 4].c();
            let (greedy, _) = greedy_layout(&layout, false);
            assert_eq!(greedy, [4, 3, 2].f());
            let (greedy, _) = greedy_layout(&layout, true);
            assert_eq!(greedy, [4, 3, 2].f());
            // f-contiguous layout
            let layout = [2, 3, 4].f();
            let (greedy, _) = greedy_layout(&layout, false);
            assert_eq!(greedy, [2, 3, 4].f());
            let (greedy, _) = greedy_layout(&layout, true);
            assert_eq!(greedy, [2, 3, 4].f());
            // dimension-size 1 or stride-size 0
            let layout = Layout::new_unchecked([5, 1, 2, 1, 3, 6], [1000, 10, 10, 40, 0, 100], 0);
            let (greedy, _) = greedy_layout(&layout, false);
            let expect = Layout::new_unchecked([2, 6, 5, 1, 1, 1], [10, 100, 1000, 0, 0, 0], 0);
            assert_eq!(greedy, expect);
            let (greedy, _) = greedy_layout(&layout, true);
            let expect = Layout::new_unchecked([1, 1, 3, 2, 6, 5], [10, 40, 0, 10, 100, 1000], 0);
            assert_eq!(greedy, expect);
            // negative strides
            let layout = [2, 3, 4].f().dim_narrow(1, slice!(None, None, -1)).unwrap();
            let layout = layout.swapaxes(-1, -2).unwrap();
            let (greedy, _) = greedy_layout(&layout, true);
            assert_eq!(greedy, [2, 3, 4].f());
            let (greedy, _) = greedy_layout(&layout, false);
            assert_eq!(greedy, [2, 3, 4].f().dim_narrow(1, slice!(None, None, -1)).unwrap());
        }
    }
}
