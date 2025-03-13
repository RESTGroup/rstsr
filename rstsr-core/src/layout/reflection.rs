//! Layout/Indexing/Slicing reflection (inverse mapping) utilities.
//!
//! Note this name does not mean reflection (反射) of program. It just means the
//! inverse (逆 ~ 反) mapping (射).

use crate::prelude_dev::*;

/// Deduce the n-dimensional index from the offset in the layout.
///
/// Tensor indexing is passing the index to layout, and returns the offset (as
/// memory shift to some anchor).
///
/// If offset is obtained by indexing the layout:
/// `offset = layout.index(index)`
/// Then this function tries to obtain the inverse mapping `index` from
/// `layout`.
///
/// # Note
///
/// The inverse mapping may not be unique.
///
/// Restrictions:
/// - Layout should not be broadcasted.
///
/// This function should not be applied in computationally intensive part.
pub fn layour_reflect_index(layout: &Layout<IxD>, offset: usize) -> Result<IxD> {
    layout.check_strides()?;
    layout.bounds_index()?;
    if layout.is_broadcasted() {
        rstsr_raise!(InvalidLayout, "Layout should not be broadcasted in `layour_reflect_index`.")?;
    }

    // 1. Prepare the result
    let mut index: Vec<usize> = vec![0; layout.ndim()];

    // 2. Get the location vector of strides, with the largest absolute stride to be
    //    the first Also, only shape != 1 leaves, other cases are ignored.
    let arg_stride = layout
        .stride()
        .iter()
        .enumerate()
        .filter(|a| layout.shape()[a.0] != 1)
        .sorted_by(|a, b| b.1.abs().cmp(&a.1.abs()))
        .collect_vec();

    // 3. Calculate the index
    let mut inner_offset = offset as isize - layout.offset() as isize;
    for (n, &(i, &s)) in arg_stride.iter().enumerate() {
        let q = inner_offset.unsigned_abs() / s.unsigned_abs();
        let r = inner_offset.unsigned_abs() % s.unsigned_abs();
        // we can not tolarate if `q != 0` and sign of `inner_offset` is not the same to
        // stride `s`. If this happens, index is negative, which is not allowed.
        if q != 0 && inner_offset * s < 0 {
            rstsr_raise!(InvalidValue, "Negative index occured.")?;
        }
        // offset is divisible by stride
        // then the leaving index should be 0, so early break
        if r == 0 {
            index[i] = q;
            break;
        }
        // if last element not divisible by stride
        // then the provided offset is invalid
        if n == arg_stride.len() - 1 {
            rstsr_raise!(InvalidValue, "Offset is not divisible by the smallest stride.")?;
        }
        // generate next inner_offset
        // next inner_offset must have the same sign with the next stride
        // so the `q` given by modular division may have to increase by 1
        inner_offset -= s * q as isize;
        index[i] = q;
        if inner_offset.is_negative() != arg_stride[n + 1].1.is_negative() {
            inner_offset -= s;
            index[i] += 1;
        }
    }

    // 4. Before return the index, we should check if the given index can recover
    //    the input offset.
    let offset_recover = layout.index_f(&index.iter().map(|&i| i as isize).collect_vec())?;
    if offset_recover != offset {
        rstsr_raise!(
            RuntimeError,
            "The given offset can not be recovered by the index, may be an internal bug."
        )?;
    }

    return Ok(index);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_layour_reflect_index() {
        let layout = vec![10, 12, 1, 15].f().dim_narrow(1, slice!(10, 2, -1)).unwrap();
        println!("{:?}", layout);

        // test usual case
        let offset = layout.index(&[2, 3, 0, 5]);
        let index = layour_reflect_index(&layout, offset).unwrap();
        assert_eq!(index, [2, 3, 0, 5]);

        // test another usual case
        let offset = layout.index(&[2, 1, 0, 0]);
        let index = layour_reflect_index(&layout, offset).unwrap();
        assert_eq!(index, [2, 1, 0, 0]);

        // test early stop case
        let offset = layout.index(&[0, 0, 0, 3]);
        let index = layour_reflect_index(&layout, offset).unwrap();
        assert_eq!(index, [0, 0, 0, 3]);

        // test case should failed (dim-1 out of bound)
        let offset = [10, 12, 1, 15].f().index(&[2, 11, 0, 3]);
        let index = layour_reflect_index(&layout, offset);
        assert!(index.is_err());
    }
}
