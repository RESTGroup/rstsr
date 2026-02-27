//! Auxiliary function for reshaping a tensor.

use crate::prelude_dev::*;

/// Attempt to reshape an array without copying data.
///
/// This is direct translation (by AI) of `_attempt_nocopy_reshape` in NumPy.
#[allow(clippy::needless_range_loop)]
fn attempt_nocopy_reshape(
    old_dims: &[usize],
    old_strides: &[isize],
    newdims: &[usize],
    is_f_order: bool,
) -> Option<Vec<isize>> {
    let mut oldnd = 0;
    let mut olddims = vec![0; old_dims.len()];
    let mut oldstrides = vec![0; old_strides.len()];
    let mut newstrides = vec![0; newdims.len()];
    let newnd = newdims.len();

    // Remove axes with dimension 1 from the old array
    for i in 0..old_dims.len() {
        if old_dims[i] != 1 {
            olddims[oldnd] = old_dims[i];
            oldstrides[oldnd] = old_strides[i];
            oldnd += 1;
        }
    }

    // oi to oj and ni to nj give the axis ranges currently worked with
    let mut oi = 0;
    let mut oj = 1;
    let mut ni = 0;
    let mut nj = 1;

    while ni < newnd && oi < oldnd {
        let mut np = newdims[ni];
        let mut op = olddims[oi];

        while np != op {
            if np < op {
                // Misses trailing 1s, these are handled later
                np *= newdims[nj];
                nj += 1;
            } else {
                op *= olddims[oj];
                oj += 1;
            }
        }

        // Check whether the original axes can be combined
        for ok in oi..(oj - 1) {
            if is_f_order {
                // Fortran order check
                if oldstrides[ok + 1] != olddims[ok] as isize * oldstrides[ok] {
                    return None; // not contiguous enough
                }
            } else {
                // C order check
                if oldstrides[ok] != olddims[ok + 1] as isize * oldstrides[ok + 1] {
                    return None; // not contiguous enough
                }
            }
        }

        // Calculate new strides for all axes currently worked with
        if is_f_order {
            newstrides[ni] = oldstrides[oi];
            for nk in (ni + 1)..nj {
                newstrides[nk] = newstrides[nk - 1] * newdims[nk - 1] as isize;
            }
        } else {
            // C order
            newstrides[nj - 1] = oldstrides[oj - 1];
            for nk in ((ni + 1)..nj).rev() {
                newstrides[nk - 1] = newstrides[nk] * newdims[nk] as isize;
            }
        }

        ni = nj;
        nj += 1;
        oi = oj;
        oj += 1;
    }

    // Set strides corresponding to trailing 1s of the new shape
    let last_stride = if ni >= 1 {
        let mut stride = newstrides[ni - 1];
        if is_f_order {
            stride *= newdims[ni - 1] as isize;
        }
        stride
    } else {
        1 // Assuming element size of 1 (PyArray_ITEMSIZE equivalent)
          // In practice, you might want to pass the item size as a parameter
    };

    for nk in ni..newnd {
        newstrides[nk] = last_stride;
    }

    Some(newstrides)
}

/// Check `-1` in shape and substitute it with the correct value.
///
/// # Arguments
/// * `shape_out` - The shape of the tensor to be reshaped.
/// * `size_known` - The size of the original tensor.
pub fn reshape_substitute_negatives(shape_out: &[isize], size_in: usize) -> Result<Vec<usize>> {
    let mut shape = shape_out.to_vec();

    // check negative indexes
    let mut idx_neg1: Option<usize> = None;
    for (i, &v) in shape.iter().enumerate() {
        match v {
            -1 => match idx_neg1 {
                Some(_) => rstsr_raise!(InvalidValue, "Only one -1 is allowed in shape.")?,
                None => idx_neg1 = Some(i),
            },
            ..-1 => {
                rstsr_raise!(InvalidValue, "Negative index must be -1.")?;
            },
            _ => (),
        }
    }

    // substitute negative index
    if let Some(idx_neg1) = idx_neg1 {
        let size_in = size_in as isize;
        let size_neg = shape.iter().fold(1, |acc, &v| if v == -1 { acc } else { acc * v });
        rstsr_assert!(
            size_in % size_neg == 0,
            InvalidValue,
            "Shape '-1' in {:?} could not be determined to original tensor size {:?}",
            shape,
            size_in
        )?;
        shape[idx_neg1] = size_in / size_neg;
    }
    return Ok(shape.iter().map(|&v| v as usize).collect::<Vec<usize>>());
}

/// A quick check for reshaping a tensor.
///
/// - check if size is the same (raise if failed)
/// - check if exactly same shape (return if true)
/// - check if contiguous (return if true)
///
/// For more complex reshaping, return `None`, and other functions should handle
/// this kind of situation.
///
/// For order option, row-major and col-major behaves differently.
fn quick_check(shape_out: &Vec<usize>, layout_in: &Layout<IxD>, order: FlagOrder) -> Result<Option<Layout<IxD>>> {
    // check if size is the same
    let size_in = layout_in.size();
    let size_out = shape_out.iter().product();
    rstsr_assert_eq!(size_in, size_out, InvalidValue, "Size mismatch between input tensor and output tensor.",)?;

    // if size is zero or one, return immediately
    // currently, we use broadcast way to handle this case
    // strides will be set to zeros, which should not affect computation
    if size_in == 0 || size_in == 1 {
        let strides = vec![0; shape_out.len()];
        return Ok(Some(Layout::<IxD>::new(shape_out.clone(), strides, layout_in.offset())?));
    }

    // check if exactly same shape
    if shape_out == layout_in.shape() {
        return Ok(Some(layout_in.clone()));
    }

    // check if contiguous
    match order {
        RowMajor => {
            if layout_in.c_contig() {
                return Ok(Some(shape_out.new_c_contig(Some(layout_in.offset()))));
            }
        },
        ColMajor => {
            if layout_in.f_contig() {
                return Ok(Some(shape_out.new_f_contig(Some(layout_in.offset()))));
            }
        },
    };

    // all easy cases checked, return None for further reshaping
    return Ok(None);
}

/// Check if a tensor can be reshaped to a new shape without explicitly copying
/// underlying data.
///
/// - If shape not match, this function will raise error.
/// - If shape match but data need to be copied, return `Ok(None)`.
/// - If everything is fine, return `Ok(Some(layout_out))`.
///
/// For order, row-major and col-major behaves differently.
pub fn layout_reshapeable(
    layout_in: &Layout<IxD>,
    shape_out: &Vec<usize>,
    order: FlagOrder,
) -> Result<Option<Layout<IxD>>> {
    Ok(quick_check(shape_out, layout_in, order)?.or_else(|| {
        attempt_nocopy_reshape(layout_in.shape(), layout_in.stride(), shape_out, order == ColMajor).map(
            |stride_out| unsafe { Layout::<IxD>::new_unchecked(shape_out.to_vec(), stride_out, layout_in.offset()) },
        )
    }))
}
