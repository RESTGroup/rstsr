//! Layout (double-ended) iterator.

use crate::prelude_dev::*;

/* #region col-major */

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
/// feature `f_prefer`.
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

impl<D> IterSplitAtAPI for IterLayoutColMajor<D>
where
    D: DimDevAPI,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        Self::split_at(&self, index).unwrap()
    }
}

/* #endregion */

/* #region row-major */

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

impl<D> IterSplitAtAPI for IterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        Self::split_at(&self, index).unwrap()
    }
}

/* #endregion */

/* #region enum of layout iterator */

#[derive(Clone, Debug)]
pub enum IterLayout<D>
where
    D: DimDevAPI,
{
    RowMajor(IterLayoutRowMajor<D>),
    ColMajor(IterLayoutColMajor<D>),
}

impl<D> IterLayout<D>
where
    D: DimDevAPI,
{
    pub fn new(layout: &Layout<D>, order: TensorIterOrder) -> Result<Self> {
        use TensorIterOrder::*;
        match order {
            C => {
                let iter = IterLayoutRowMajor::new(layout)?;
                return Ok(Self::RowMajor(iter));
            },
            F => {
                let iter = IterLayoutColMajor::new(layout)?;
                return Ok(Self::ColMajor(iter));
            },
            A => match TensorOrder::default() {
                TensorOrder::C => {
                    let iter = IterLayoutRowMajor::new(layout)?;
                    return Ok(Self::RowMajor(iter));
                },
                TensorOrder::F => {
                    let iter = IterLayoutColMajor::new(layout)?;
                    return Ok(Self::ColMajor(iter));
                },
            },
            K | G => {
                let layout = translate_to_col_major_unary(layout, TensorIterOrder::K)?;
                let iter = IterLayoutColMajor::new(&layout)?;
                return Ok(Self::ColMajor(iter));
            },
            _ => rstsr_raise!(InvalidValue),
        }
    }
}

impl<D> Iterator for IterLayout<D>
where
    D: DimDevAPI,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::RowMajor(iter) => iter.next(),
            Self::ColMajor(iter) => iter.next(),
        }
    }
}

impl<D> DoubleEndedIterator for IterLayout<D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::RowMajor(iter) => iter.next_back(),
            Self::ColMajor(iter) => iter.next_back(),
        }
    }
}

impl<D> ExactSizeIterator for IterLayout<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        match self {
            Self::RowMajor(iter) => iter.len(),
            Self::ColMajor(iter) => iter.len(),
        }
    }
}

impl<D> IterSplitAtAPI for IterLayout<D>
where
    D: DimDevAPI,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        match self {
            Self::RowMajor(iter) => {
                let (lhs, rhs) = iter.split_at(index);
                (Self::RowMajor(lhs), Self::RowMajor(rhs))
            },
            Self::ColMajor(iter) => {
                let (lhs, rhs) = iter.split_at(index);
                (Self::ColMajor(lhs), Self::ColMajor(rhs))
            },
        }
    }
}

/* #endregion */

/* #region layout iterator with index */

#[derive(Clone, Debug)]
pub struct IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    pub(crate) layout_iter: IterLayout<D>,
}

impl<D> IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    pub fn new(layout: &Layout<D>, order: FlagOrder) -> Result<Self> {
        let order = match order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        Ok(Self { layout_iter: IterLayout::new(layout, order)? })
    }
}

impl<D> Iterator for IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    type Item = (D, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.layout_iter.next().map(|offset| (index, offset))
    }
}

impl<D> DoubleEndedIterator for IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = match &self.layout_iter {
            IterLayout::ColMajor(iter_inner) => iter_inner.index_start.clone(),
            IterLayout::RowMajor(iter_inner) => iter_inner.index_start.clone(),
        };
        self.layout_iter.next_back().map(|offset| (index, offset))
    }
}

impl<D> ExactSizeIterator for IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.layout_iter.len()
    }
}

impl<D> IterSplitAtAPI for IndexedIterLayout<D>
where
    D: DimDevAPI,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (lhs, rhs) = self.layout_iter.split_at(mid);
        let lhs = IndexedIterLayout { layout_iter: lhs };
        let rhs = IndexedIterLayout { layout_iter: rhs };
        (lhs, rhs)
    }
}

/* #endregion */

/* #region col-major layout dim dispatch */

#[allow(unused_mut)]
pub fn layout_col_major_dim_dispatch_1<D, F>(la: &Layout<D>, mut f: F) -> Result<()>
where
    D: DimAPI,
    F: FnMut(usize),
{
    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                iter_a.for_each(f);
            }};
        }
        match la.ndim() {
            0 => f(la.offset()),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                iter_a.for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        iter_a.for_each(f);
    }
    Ok(())
}

#[allow(unused_mut)]
pub fn layout_col_major_dim_dispatch_2<D, F>(la: &Layout<D>, lb: &Layout<D>, mut f: F) -> Result<()>
where
    D: DimAPI,
    F: FnMut((usize, usize)),
{
    rstsr_assert_eq!(la.ndim(), lb.ndim(), RuntimeError)?;

    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dim>()?)?;
                izip!(iter_a, iter_b).for_each(f);
            }};
        }
        match la.ndim() {
            0 => f((la.offset(), lb.offset())),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                izip!(iter_a, iter_b).for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        izip!(iter_a, iter_b).for_each(f);
    }
    Ok(())
}

#[allow(unused_mut)]
pub fn layout_col_major_dim_dispatch_3<D, F>(
    la: &Layout<D>,
    lb: &Layout<D>,
    lc: &Layout<D>,
    mut f: F,
) -> Result<()>
where
    D: DimAPI,
    F: FnMut((usize, usize, usize)),
{
    rstsr_assert_eq!(la.ndim(), lb.ndim(), RuntimeError)?;
    rstsr_assert_eq!(la.ndim(), lc.ndim(), RuntimeError)?;

    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dim: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dim>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dim>()?)?;
                let iter_c = IterLayoutColMajor::new(&lc.to_dim::<$dim>()?)?;
                izip!(iter_a, iter_b, iter_c).for_each(f);
            }};
        }
        match la.ndim() {
            0 => f((la.offset(), lb.offset(), lc.offset())),
            1 => dispatch!(Ix1),
            2 => dispatch!(Ix2),
            3 => dispatch!(Ix3),
            4 => dispatch!(Ix4),
            5 => dispatch!(Ix5),
            6 => dispatch!(Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                let iter_c = IterLayoutColMajor::new(lc)?;
                izip!(iter_a, iter_b, iter_c).for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        let iter_c = IterLayoutColMajor::new(lc)?;
        izip!(iter_a, iter_b, iter_c).for_each(f);
    }
    Ok(())
}

#[allow(unused_mut)]
pub fn layout_col_major_dim_dispatch_2diff<DA, DB, F>(
    la: &Layout<DA>,
    lb: &Layout<DB>,
    mut f: F,
) -> Result<()>
where
    DA: DimAPI,
    DB: DimAPI,
    F: FnMut((usize, usize)),
{
    #[cfg(feature = "dispatch_dim_layout_iter")]
    {
        macro_rules! dispatch {
            ($dima: ident, $dimb: ident) => {{
                let iter_a = IterLayoutColMajor::new(&la.to_dim::<$dima>()?)?;
                let iter_b = IterLayoutColMajor::new(&lb.to_dim::<$dimb>()?)?;
                izip!(iter_a, iter_b).for_each(f);
            }};
        }
        match (la.ndim(), lb.ndim()) {
            (0, 0) => f((la.offset(), lb.offset())),
            (1, 1) => dispatch!(Ix1, Ix1),
            (1, 2) => dispatch!(Ix1, Ix2),
            (1, 3) => dispatch!(Ix1, Ix3),
            (1, 4) => dispatch!(Ix1, Ix4),
            (1, 5) => dispatch!(Ix1, Ix5),
            (1, 6) => dispatch!(Ix1, Ix6),
            (2, 1) => dispatch!(Ix2, Ix1),
            (2, 2) => dispatch!(Ix2, Ix2),
            (2, 3) => dispatch!(Ix2, Ix3),
            (2, 4) => dispatch!(Ix2, Ix4),
            (2, 5) => dispatch!(Ix2, Ix5),
            (2, 6) => dispatch!(Ix2, Ix6),
            (3, 1) => dispatch!(Ix3, Ix1),
            (3, 2) => dispatch!(Ix3, Ix2),
            (3, 3) => dispatch!(Ix3, Ix3),
            (3, 4) => dispatch!(Ix3, Ix4),
            (3, 5) => dispatch!(Ix3, Ix5),
            (3, 6) => dispatch!(Ix3, Ix6),
            (4, 1) => dispatch!(Ix4, Ix1),
            (4, 2) => dispatch!(Ix4, Ix2),
            (4, 3) => dispatch!(Ix4, Ix3),
            (4, 4) => dispatch!(Ix4, Ix4),
            (4, 5) => dispatch!(Ix4, Ix5),
            (4, 6) => dispatch!(Ix4, Ix6),
            (5, 1) => dispatch!(Ix5, Ix1),
            (5, 2) => dispatch!(Ix5, Ix2),
            (5, 3) => dispatch!(Ix5, Ix3),
            (5, 4) => dispatch!(Ix5, Ix4),
            (5, 5) => dispatch!(Ix5, Ix5),
            (5, 6) => dispatch!(Ix5, Ix6),
            (6, 1) => dispatch!(Ix6, Ix1),
            (6, 2) => dispatch!(Ix6, Ix2),
            (6, 3) => dispatch!(Ix6, Ix3),
            (6, 4) => dispatch!(Ix6, Ix4),
            (6, 5) => dispatch!(Ix6, Ix5),
            (6, 6) => dispatch!(Ix6, Ix6),
            _ => {
                let iter_a = IterLayoutColMajor::new(la)?;
                let iter_b = IterLayoutColMajor::new(lb)?;
                izip!(iter_a, iter_b).for_each(f);
            },
        }
    }

    #[cfg(not(feature = "dispatch_dim_layout_iter"))]
    {
        let iter_a = IterLayoutColMajor::new(la)?;
        let iter_b = IterLayoutColMajor::new(lb)?;
        izip!(iter_a, iter_b).for_each(f);
    }
    Ok(())
}

/* #endregion */

#[cfg(test)]
mod test_col_major {
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
        println!("{:?}", unsafe { layout.shape().unravel_index_f(100) });
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

#[cfg(test)]
mod test_row_major {
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
