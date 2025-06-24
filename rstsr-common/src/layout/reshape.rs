//! Auxiliary function for reshaping a tensor.

use crate::prelude_dev::*;

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

/// Internal function that pop input layout.
///
/// This function is for c-prefer (row-major) only.
///
/// # Returns
/// * `Vec<usize>` - The size of partly contiguous (with a minimum stride) batch of input tensor.
/// * `Vec<isize>` - The minimum stride of the current batch.
fn pop_layout_in(shape_in: &mut Vec<usize>, stride_in: &mut Vec<isize>) -> (usize, isize) {
    rstsr_assert_eq!(shape_in.len(), stride_in.len(), RuntimeError).unwrap();
    rstsr_assert!(!shape_in.is_empty(), RuntimeError).unwrap();

    let mut stride_min = stride_in.pop().unwrap();
    let mut size = shape_in.pop().unwrap();

    // determine if current batch is broadcasted
    if size == 1 || stride_min == 0 {
        // broadcasted, reset stride_min to 0
        stride_min = 0;
        while stride_in.last().is_some_and(|&v| v == 0) || shape_in.last().is_some_and(|&v| v == 1) {
            stride_in.pop();
            size *= shape_in.pop().unwrap();
        }
        return (size, stride_min);
    } else {
        // general case
        while stride_in.last().is_some_and(|&v| v == size as isize * stride_min) {
            stride_in.pop();
            size *= shape_in.pop().unwrap();
        }
        return (size, stride_min);
    }
}

/// Internal function that pop output shape, and inject output strides.
///
/// This function is for c-prefer (row-major) only.
/// However, note that `stride_out` is in reverse order.
///
/// This function will return `true/false` depending on compatibility of shape.
fn pop_shape_out(
    shape_out: &mut Vec<usize>,
    stride_out: &mut Vec<isize>,
    mut size: usize,
    mut stride_min: isize,
) -> bool {
    rstsr_assert!(!shape_out.is_empty(), RuntimeError).unwrap();

    while size != 1 || shape_out.last().is_some_and(|&v| v == 1) {
        let s_out = shape_out.pop().unwrap();
        if size % s_out != 0 {
            return false;
        }
        size /= s_out;
        stride_out.push(stride_min);
        stride_min *= s_out as isize;
    }

    return true;
}

/// Internal function for reshaping a tensor in any cases.
fn complicated_reshape(shape_out: &[usize], layout_in: &Layout<IxD>, order: FlagOrder) -> Option<Layout<IxD>> {
    let shape_out_ref = shape_out; // the original shape_out not modified
    let mut shape_out = shape_out.to_vec(); // the shape_out to be destroyed in iteration
    let mut stride_out = Vec::new();
    let mut shape_in = layout_in.shape().to_vec();
    let mut stride_in = layout_in.stride().to_vec();
    let offset = layout_in.offset();

    // f-prefer handled by reversing everything
    if order == FlagOrder::F {
        shape_in.reverse();
        stride_in.reverse();
        shape_out.reverse();
    }

    while !shape_in.is_empty() {
        let (size_in, stride_in_min) = pop_layout_in(&mut shape_in, &mut stride_in);
        if !pop_shape_out(&mut shape_out, &mut stride_out, size_in, stride_in_min) {
            return None;
        }
    }
    rstsr_assert!(shape_out.is_empty(), RuntimeError).unwrap();
    rstsr_assert_eq!(stride_out.len(), shape_out_ref.len(), RuntimeError).unwrap();
    // note that stride_out is in reverse order in c-prefer
    // as contrary, shape_out is in reverse order in f-prefer
    match order {
        RowMajor => stride_out.reverse(),
        ColMajor => shape_out.reverse(),
    };

    let layout_out = unsafe { Layout::<IxD>::new_unchecked(shape_out_ref.to_vec(), stride_out, offset) };
    return Some(layout_out);
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
    if let Some(layout_out) = quick_check(shape_out, layout_in, order)? {
        return Ok(Some(layout_out));
    }
    return Ok(complicated_reshape(shape_out, layout_in, order));
}
