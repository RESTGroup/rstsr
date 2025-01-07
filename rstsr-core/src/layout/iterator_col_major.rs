//! Layout (double-ended) iterator; only column-major iterator is implemented.

use crate::prelude_dev::*;

/// Layout iterator (column-major).
///
/// This iterator only handles column-major iterator.
/// For other iteration orders, use function [`translate_to_col_major`] to
/// generate the corresponding col-major (f-prefer) layout, then iterate as
/// col-major.
///
/// # Note
///
/// This crate implements col-major iterator only; the layout iterator that
/// actaully works is internal realization; though it's public struct, it is not
/// intended to be exposed to user.
/// Choosing col-major iterator is because it is possibly the most efficient
/// way. It is not related to default order, which could be defined by crate
/// feature `c_prefer`.
#[derive(Clone, Debug)]
pub struct IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,

    pub(crate) index_start: D, // this is not used for buffer-order
    pub(crate) iter_start: usize,
    pub(crate) offset_start: isize,

    pub(crate) index_end: D, // this is not used for buffer-order
    pub(crate) iter_end: usize,
    pub(crate) offset_end: isize,
}

impl<D> IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    /// This function generates col-major (f-prefer) layout, then give its
    /// iterator object.
    pub fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let shape = layout.shape();
        let iter_start = 0;
        let iter_end = layout.size();
        let index_start = layout.new_shape();
        let index_end = unsafe { shape.unravel_index_f(iter_end) };
        let offset_start = layout.offset() as isize;
        let offset_end = unsafe { layout.index_uncheck(index_end.as_ref()) };

        return Ok(Self {
            layout,
            index_start,
            iter_start,
            offset_start,
            index_end,
            iter_end,
            offset_end,
        });
    }

    pub fn split_at(&self, index: usize) -> Result<(Self, Self)> {
        let Self { layout, index_start, iter_start, offset_start, index_end, iter_end, offset_end } =
            self.clone();
        let shape = layout.shape();
        let iter_ins = iter_start + index;
        let index_ins = unsafe { shape.unravel_index_f(iter_ins) };
        let offset_ins = unsafe { layout.index_uncheck(index_ins.as_ref()) };
        let split_lhs = Self {
            layout: layout.clone(),
            index_start,
            iter_start,
            offset_start,
            index_end: index_ins.clone(),
            iter_end: iter_ins,
            offset_end: offset_ins,
        };
        let split_rhs = Self {
            layout: layout.clone(),
            index_start: index_ins,
            iter_start: iter_ins,
            offset_start: offset_ins,
            index_end,
            iter_end,
            offset_end,
        };
        return Ok((split_lhs, split_rhs));
    }
}

impl<D> IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    #[inline]
    fn next_iter_index(&mut self) {
        let layout = &self.layout;
        let index = self.index_start.as_mut();
        let mut offset = self.offset_start;
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match layout.ndim() {
            0 => (),
            1 => {
                index[0] += 1;
                offset += stride[0];
            },
            2 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                }
            },
            3 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                    if index[1] == shape[1] {
                        index[1] = 0;
                        offset -= shape[1] as isize * stride[1];
                        index[2] += 1;
                        offset += stride[2];
                    }
                }
            },
            4 => {
                index[0] += 1;
                offset += stride[0];
                if index[0] == shape[0] {
                    index[0] = 0;
                    offset -= shape[0] as isize * stride[0];
                    index[1] += 1;
                    offset += stride[1];
                    if index[1] == shape[1] {
                        index[1] = 0;
                        offset -= shape[1] as isize * stride[1];
                        index[2] += 1;
                        offset += stride[2];
                        if index[2] == shape[2] {
                            index[2] = 0;
                            offset -= shape[2] as isize * stride[2];
                            index[3] += 1;
                            offset += stride[3];
                        }
                    }
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut()) {
                    *idx += 1;
                    offset += t;
                    if idx == d {
                        *idx = 0;
                        offset -= *d as isize * t;
                    } else {
                        break;
                    }
                }
            },
        }
        self.offset_start = offset;
        self.iter_start += 1;
    }

    #[inline]
    fn back_iter_index(&mut self) {
        let layout = &self.layout;
        let index = self.index_end.as_mut();
        let mut offset = self.offset_end;
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match layout.ndim() {
            0 => (),
            1 => {
                index[0] -= 1;
                offset -= stride[0];
            },
            2 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    index[1] -= 1;
                    offset -= stride[1];
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            3 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    if index[1] == 0 {
                        index[1] = shape[1] - 1;
                        offset += (shape[1] - 1) as isize * stride[1];
                        index[2] -= 1;
                        offset -= stride[2];
                    } else {
                        index[1] -= 1;
                        offset -= stride[1];
                    }
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            4 => {
                if index[0] == 0 {
                    index[0] = shape[0] - 1;
                    offset += (shape[0] - 1) as isize * stride[0];
                    if index[1] == 0 {
                        index[1] = shape[1] - 1;
                        offset += (shape[1] - 1) as isize * stride[1];
                        if index[2] == 0 {
                            index[2] = shape[2] - 1;
                            offset += (shape[2] - 1) as isize * stride[2];
                            index[3] -= 1;
                            offset -= stride[3];
                        } else {
                            index[2] -= 1;
                            offset -= stride[2];
                        }
                    } else {
                        index[1] -= 1;
                        offset -= stride[1];
                    }
                } else {
                    index[0] -= 1;
                    offset -= stride[0];
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut()) {
                    if *idx == 0 {
                        *idx = *d - 1;
                        offset += (*d - 1) as isize * t;
                    } else {
                        *idx -= 1;
                        offset -= t;
                        break;
                    }
                }
            },
        }
        self.offset_end = offset;
        self.iter_end -= 1;
    }
}

impl<D> Iterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_start >= self.iter_end {
            return None;
        }
        let offset = self.offset_start;
        self.next_iter_index();
        return Some(offset.try_into().unwrap());
    }
}

impl<D> DoubleEndedIterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.iter_start >= self.iter_end {
            return None;
        }
        self.back_iter_index();
        let offset = self.offset_end;
        return Some(offset.try_into().unwrap());
    }
}

impl<D> ExactSizeIterator for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.iter_end - self.iter_start
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // type alias for this file
    type Order = TensorIterOrder;

    #[test]
    fn test_iter_next() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        // a = np.asfortranarray(a)
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            782, 797, 812, 827, 842, 857, 602, 617, 632, 647, 662, 677, 785, 800, 815, 830, 845,
            860, 605, 620, 635, 650, 665, 680, 788, 803, 818, 833, 848, 863, 608, 623, 638, 653,
            668, 683
        ]);
        // np.array(np.nditer(a, order="F"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::F).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            782, 785, 788, 602, 605, 608, 797, 800, 803, 617, 620, 623, 812, 815, 818, 632, 635,
            638, 827, 830, 833, 647, 650, 653, 842, 845, 848, 662, 665, 668, 857, 860, 863, 677,
            680, 683
        ]);
        // np.array(np.nditer(a, order="K"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::K).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            602, 605, 608, 617, 620, 623, 632, 635, 638, 647, 650, 653, 662, 665, 668, 677, 680,
            683, 782, 785, 788, 797, 800, 803, 812, 815, 818, 827, 830, 833, 842, 845, 848, 857,
            860, 863
        ]);
        // np.array(np.nditer(a, order="G"))
        // for no broadcast case, greedy-order is same as k-order
        let layout_trans = translate_to_col_major_unary(&layout, Order::K).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            602, 605, 608, 617, 620, 623, 632, 635, 638, 647, 650, 653, 662, 665, 668, 677, 680,
            683, 782, 785, 788, 797, 800, 803, 812, 815, 818, 827, 830, 833, 842, 845, 848, 857,
            860, 863
        ]);
        // buffer should fail
        assert!(translate_to_col_major_unary(&layout, Order::B).is_err());
    }

    #[test]
    fn test_iter_back() {
        let layout = Layout::new([10, 10, 10], [10, 1, 100], 0).unwrap();
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        println!("{:?}", unsafe { layout.unravel_index_f(100) });
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.collect::<Vec<_>>();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_back = iter.rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_back_empty() {
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.clone().collect::<Vec<_>>();
        let vec_back = iter.clone().rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());

        let layout = Layout::new([10], [10], 10).unwrap();
        // np.array(np.nditer(a, order="C"))
        let layout_trans = translate_to_col_major_unary(&layout, Order::C).unwrap();
        let iter = IterLayoutColMajor::new(&layout_trans).unwrap();
        let vec_next = iter.clone().collect::<Vec<_>>();
        let vec_back = iter.clone().rev().collect::<Vec<_>>();
        assert_eq!(vec_next, vec_back.iter().rev().copied().collect::<Vec<_>>());
    }
}
