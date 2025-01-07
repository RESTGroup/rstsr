//! Layout (double-ended) iterator; only row-major iterator is implemented.

use crate::prelude_dev::*;

/// Layout iterator (row-major).
///
/// This iterator only handles row-major iterator.
///
/// # Note
///
/// This crate implements row-major iterator only; the layout iterator that
/// actaully works is internal realization; though it's public struct, it is not
/// intended to be exposed to user.
#[derive(Debug, Clone)]
pub struct IterLayoutRowMajor<D>
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

impl<D> IterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    /// This function generates row-major (c-prefer) layout, then give its
    /// iterator object.
    pub fn new(layout: &Layout<D>) -> Result<Self> {
        let layout = layout.clone();
        let shape = layout.shape();
        let iter_start = 0;
        let iter_end = layout.size();
        let index_start = layout.new_shape();
        let index_end = unsafe { shape.unravel_index_c(iter_end) };
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
        let index_ins = unsafe { shape.unravel_index_c(iter_ins) };
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

impl<D> IterLayoutRowMajor<D>
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
                index[1] += 1;
                offset += stride[1];
                if index[1] == shape[1] {
                    index[1] = 0;
                    offset -= shape[1] as isize * stride[1];
                    index[0] += 1;
                    offset += stride[0];
                }
            },
            3 => {
                index[2] += 1;
                offset += stride[2];
                if index[2] == shape[2] {
                    index[2] = 0;
                    offset -= shape[2] as isize * stride[2];
                    index[1] += 1;
                    offset += stride[1];
                    if index[1] == shape[1] {
                        index[1] = 0;
                        offset -= shape[1] as isize * stride[1];
                        index[0] += 1;
                        offset += stride[0];
                    }
                }
            },
            4 => {
                index[3] += 1;
                offset += stride[3];
                if index[3] == shape[3] {
                    index[3] = 0;
                    offset -= shape[3] as isize * stride[3];
                    index[2] += 1;
                    offset += stride[2];
                    if index[2] == shape[2] {
                        index[2] = 0;
                        offset -= shape[2] as isize * stride[2];
                        index[1] += 1;
                        offset += stride[1];
                        if index[1] == shape[1] {
                            index[1] = 0;
                            offset -= shape[1] as isize * stride[1];
                            index[0] += 1;
                            offset += stride[0];
                        }
                    }
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut()).rev() {
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
                if index[1] == 0 {
                    index[1] = shape[1] - 1;
                    offset += (shape[1] - 1) as isize * stride[1];
                    index[0] -= 1;
                    offset -= stride[0];
                } else {
                    index[1] -= 1;
                    offset -= stride[1];
                }
            },
            3 => {
                if index[2] == 0 {
                    index[2] = shape[2] - 1;
                    offset += (shape[2] - 1) as isize * stride[2];
                    if index[1] == 0 {
                        index[1] = shape[1] - 1;
                        offset += (shape[1] - 1) as isize * stride[1];
                        index[0] -= 1;
                        offset -= stride[0];
                    } else {
                        index[1] -= 1;
                        offset -= stride[1];
                    }
                } else {
                    index[2] -= 1;
                    offset -= stride[2];
                }
            },
            4 => {
                if index[3] == 0 {
                    index[3] = shape[3] - 1;
                    offset += (shape[3] - 1) as isize * stride[3];
                    if index[2] == 0 {
                        index[2] = shape[2] - 1;
                        offset += (shape[2] - 1) as isize * stride[2];
                        if index[1] == 0 {
                            index[1] = shape[1] - 1;
                            offset += (shape[1] - 1) as isize * stride[1];
                            index[0] -= 1;
                            offset -= stride[0];
                        } else {
                            index[1] -= 1;
                            offset -= stride[1];
                        }
                    } else {
                        index[2] -= 1;
                        offset -= stride[2];
                    }
                } else {
                    index[3] -= 1;
                    offset -= stride[3];
                }
            },
            _ => {
                for (d, t, idx) in izip!(shape, stride, index.as_mut()).rev() {
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

impl<D> Iterator for IterLayoutRowMajor<D>
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

impl<D> DoubleEndedIterator for IterLayoutRowMajor<D>
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

impl<D> ExactSizeIterator for IterLayoutRowMajor<D>
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

    #[test]
    fn test_iter_next() {
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        // np.array(np.nditer(a, order="C"))
        let iter = IterLayoutRowMajor::new(&layout).unwrap();
        let vec = iter.collect::<Vec<_>>();
        assert_eq!(vec, [
            782, 797, 812, 827, 842, 857, 602, 617, 632, 647, 662, 677, 785, 800, 815, 830, 845,
            860, 605, 620, 635, 650, 665, 680, 788, 803, 818, 833, 848, 863, 608, 623, 638, 653,
            668, 683
        ]);
        let iter = IterLayoutRowMajor::new(&layout).unwrap();
        let vec = iter.rev().collect::<Vec<_>>();
        assert_eq!(vec, [
            683, 668, 653, 638, 623, 608, 863, 848, 833, 818, 803, 788, 680, 665, 650, 635, 620,
            605, 860, 845, 830, 815, 800, 785, 677, 662, 647, 632, 617, 602, 857, 842, 827, 812,
            797, 782
        ]);
    }
}
