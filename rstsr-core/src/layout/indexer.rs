use crate::prelude_dev::*;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indexer {
    /// Slice the tensor by a range, denoted by slice instead of
    /// std::ops::Range.
    Slice(SliceI),
    /// Marginalize one dimension out by index.
    Select(isize),
    /// Insert dimension at index, something like unsqueeze. Currently not
    /// applied.
    Insert,
    /// Expand dimensions.
    Ellipsis,
}

impl<R> From<R> for Indexer
where
    R: Into<SliceI>,
{
    fn from(slice: R) -> Self {
        Self::Slice(slice.into())
    }
}

impl From<Option<usize>> for Indexer {
    fn from(opt: Option<usize>) -> Self {
        match opt {
            Some(_) => panic!("Option<T> should not be used in Indexer."),
            None => Self::Insert,
        }
    }
}

macro_rules! impl_from_int_into_indexer {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Indexer {
                fn from(index: $t) -> Self {
                    Self::Select(index as isize)
                }
            }
        )*
    };
}

impl_from_int_into_indexer!(usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

pub trait IndexerPreserveAPI: Sized {
    /// Narrowing tensor by slicing at a specific axis.
    fn dim_narrow(&self, axis: isize, slice: SliceI) -> Result<Self>;
}

impl<D> IndexerPreserveAPI for Layout<D>
where
    D: DimDevAPI,
{
    fn dim_narrow(&self, axis: isize, slice: SliceI) -> Result<Self> {
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        // get essential information
        let mut shape = self.shape().clone();
        let mut stride = self.stride().clone();

        // fast return if slice is empty
        if slice == (Slice { start: None, stop: None, step: None }) {
            return Ok(self.clone());
        }

        // previous shape length
        let len_prev = shape[axis] as isize;

        // handle cases of step > 0 and step < 0
        let step = slice.step().unwrap_or(1);
        rstsr_assert!(step != 0, InvalidValue)?;

        // quick return if previous shape is zero
        if len_prev == 0 {
            return Ok(self.clone());
        }

        if step > 0 {
            // default start = 0 and stop = len_prev
            let mut start = slice.start().unwrap_or(0);
            let mut stop = slice.stop().unwrap_or(len_prev);

            // handle negative slice
            if start < 0 {
                start = (len_prev + start).max(0);
            }
            if stop < 0 {
                stop = (len_prev + stop).max(0);
            }

            if start > len_prev || start > stop {
                // zero size slice caused by inproper start and stop
                start = 0;
                stop = 0;
            } else if stop > len_prev {
                // stop is out of bound, set it to len_prev
                stop = len_prev;
            }

            let offset = (self.offset() as isize + stride[axis] * start) as usize;
            shape[axis] = ((stop - start + step - 1) / step).max(0) as usize;
            stride[axis] *= step;
            return Ok(Self::new(shape, stride, offset));
        } else {
            // step < 0
            // default start = len_prev - 1 and stop = -1
            let mut start = slice.start().unwrap_or(len_prev - 1);
            let mut stop = slice.stop().unwrap_or(-1);

            // handle negative slice
            if start < 0 {
                start = (len_prev + start).max(0);
            }
            if stop < -1 {
                stop = (len_prev + stop).max(-1);
            }

            if stop > len_prev - 1 || stop > start {
                // zero size slice caused by inproper start and stop
                start = 0;
                stop = 0;
            } else if start > len_prev - 1 {
                // start is out of bound, set it to len_prev
                start = len_prev - 1;
            }

            let offset = (self.offset() as isize + stride[axis] * start) as usize;
            shape[axis] = ((stop - start - step - 1) / step).max(0) as usize;
            stride[axis] *= step;
            return Ok(Self::new(shape, stride, offset));
        }
    }
}

pub trait IndexerSmallerOneAPI {
    type DOut: DimDevAPI;

    /// Select dimension at index. Number of dimension will decrease by 1.
    fn dim_select(&self, axis: isize, index: isize) -> Result<Layout<Self::DOut>>;

    /// Eliminate dimension at index. Number of dimension will decrease by 1.
    fn dim_eliminate(&self, axis: isize) -> Result<Layout<Self::DOut>>;
}

impl<D> IndexerSmallerOneAPI for Layout<D>
where
    D: DimDevAPI + DimSmallerOneAPI,
    D::SmallerOne: DimDevAPI,
{
    type DOut = <D as DimSmallerOneAPI>::SmallerOne;

    fn dim_select(&self, axis: isize, index: isize) -> Result<Layout<Self::DOut>> {
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        // get essential information
        let shape = self.shape();
        let stride = self.stride();
        let mut offset = self.offset() as isize;
        let mut shape_new = vec![];
        let mut stride_new = vec![];

        // change everything
        for (i, (&d, &s)) in shape.as_ref().iter().zip(stride.as_ref().iter()).enumerate() {
            if i == axis {
                // dimension to be selected
                let idx = if index < 0 { d as isize + index } else { index };
                rstsr_pattern!(idx, 0..d as isize, ValueOutOfRange)?;
                offset += s * idx;
            } else {
                // other dimensions
                shape_new.push(d);
                stride_new.push(s);
            }
        }

        let offset = offset as usize;
        let layout = Layout::<IxD>::new(shape_new, stride_new, offset);
        return layout.into_dim();
    }

    fn dim_eliminate(&self, axis: isize) -> Result<Layout<Self::DOut>> {
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        // get essential information
        let mut shape = self.shape().as_ref().to_vec();
        let mut stride = self.stride().as_ref().to_vec();
        let offset = self.offset();

        if shape[axis] != 1 {
            rstsr_raise!(InvalidValue, "Dimension to be eliminated is not 1.")?;
        }

        shape.remove(axis);
        stride.remove(axis);

        let layout = Layout::<IxD>::new(shape, stride, offset);
        return layout.into_dim();
    }
}

pub trait IndexerLargerOneAPI {
    type DOut: DimDevAPI;

    /// Insert dimension after, with shape 1. Number of dimension will increase
    /// by 1.
    fn dim_insert(&self, axis: isize) -> Result<Layout<Self::DOut>>;
}

impl<D> IndexerLargerOneAPI for Layout<D>
where
    D: DimDevAPI + DimLargerOneAPI,
    D::LargerOne: DimDevAPI,
{
    type DOut = <D as DimLargerOneAPI>::LargerOne;

    fn dim_insert(&self, axis: isize) -> Result<Layout<Self::DOut>> {
        // dimension check
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..(self.ndim() + 1) as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        // get essential information
        let is_f_prefer = self.f_prefer();
        let mut shape = self.shape().as_ref().to_vec();
        let mut stride = self.stride().as_ref().to_vec();
        let offset = self.offset();

        if is_f_prefer {
            if axis == 0 {
                shape.insert(0, 1);
                stride.insert(0, 1);
            } else {
                shape.insert(axis, 1);
                stride.insert(axis, stride[axis - 1]);
            }
        } else if axis == self.ndim() {
            shape.push(1);
            stride.push(1);
        } else {
            shape.insert(axis, 1);
            stride.insert(axis, stride[axis]);
        }

        let layout = Layout::new(shape, stride, offset);
        return layout.into_dim();
    }
}

pub trait IndexerDynamicAPI: IndexerPreserveAPI {
    /// Index tensor by a list of indexers.
    fn dim_slice(&self, indexers: &[Indexer]) -> Result<Layout<IxD>>;

    /// Split current layout into two layouts at axis, with offset unchanged.
    fn dim_split_at(&self, axis: isize) -> Result<(Layout<IxD>, Layout<IxD>)>;
}

impl<D> IndexerDynamicAPI for Layout<D>
where
    D: DimDevAPI,
{
    fn dim_slice(&self, indexers: &[Indexer]) -> Result<Layout<IxD>> {
        // transform any layout to dynamic layout
        let shape = self.shape().as_ref().to_vec();
        let stride = self.stride().as_ref().to_vec();
        let mut layout = Layout::new(shape, stride, self.offset);

        // clone indexers to vec to make it changeable
        let mut indexers = indexers.to_vec();

        // counter for indexer
        let mut counter_slice = 0;
        let mut counter_select = 0;
        let mut idx_ellipsis = None;
        for (n, indexer) in indexers.iter().enumerate() {
            match indexer {
                Indexer::Slice(_) => counter_slice += 1,
                Indexer::Select(_) => counter_select += 1,
                Indexer::Ellipsis => match idx_ellipsis {
                    Some(_) => rstsr_raise!(InvalidValue, "Only one ellipsis indexer allowed.")?,
                    None => idx_ellipsis = Some(n),
                },
                _ => {},
            }
        }

        // check if slice-type and select-type indexer exceed the number of dimensions
        rstsr_pattern!(counter_slice + counter_select, 0..=self.ndim(), ValueOutOfRange)?;

        // insert Ellipsis by slice(:) anyway, default append at last
        let n_ellipsis = self.ndim() - counter_slice - counter_select;
        if n_ellipsis == 0 {
            if let Some(idx) = idx_ellipsis {
                indexers.remove(idx);
            }
        } else {
            let idx_ellipsis = idx_ellipsis.unwrap_or(indexers.len());
            indexers[idx_ellipsis] = SliceI { start: None, stop: None, step: None }.into();
            if n_ellipsis > 1 {
                for _ in 1..n_ellipsis {
                    indexers.insert(
                        idx_ellipsis,
                        SliceI { start: None, stop: None, step: None }.into(),
                    );
                }
            }
        }

        // handle indexers from last
        // it is possible to be zero-dim, minus after -= 1
        let mut cur_dim = self.ndim() as isize;
        for indexer in indexers.iter().rev() {
            match indexer {
                Indexer::Slice(slice) => {
                    cur_dim -= 1;
                    layout = layout.dim_narrow(cur_dim, *slice)?;
                },
                Indexer::Select(index) => {
                    cur_dim -= 1;
                    layout = layout.dim_select(cur_dim, *index)?;
                },
                Indexer::Insert => {
                    layout = layout.dim_insert(cur_dim)?;
                },
                _ => rstsr_raise!(InvalidValue, "Invalid indexer found : {:?}", indexer)?,
            }
        }

        // this program should be designed that cur_dim is zero at the end
        rstsr_assert!(cur_dim == 0, Miscellaneous, "Internal program error in indexer.")?;

        return Ok(layout);
    }

    fn dim_split_at(&self, axis: isize) -> Result<(Layout<IxD>, Layout<IxD>)> {
        // dimension check
        // this functions allows [-n, n], not previous functions [-n, n)
        let axis = if axis < 0 { self.ndim() as isize + axis } else { axis };
        rstsr_pattern!(axis, 0..=self.ndim() as isize, ValueOutOfRange)?;
        let axis = axis as usize;

        // split layouts
        let shape = self.shape().as_ref().to_vec();
        let stride = self.stride().as_ref().to_vec();
        let offset = self.offset();

        let (shape1, shape2) = shape.split_at(axis);
        let (stride1, stride2) = stride.split_at(axis);

        let layout1 = unsafe { Layout::new_unchecked(shape1.to_vec(), stride1.to_vec(), offset) };
        let layout2 = unsafe { Layout::new_unchecked(shape2.to_vec(), stride2.to_vec(), offset) };
        return Ok((layout1, layout2));
    }
}

/// Generate slice with into support and optional parameters.
#[macro_export]
macro_rules! slice {
    ($stop:expr) => {
        Slice::<isize>::from(Slice { start: None, stop: $stop.into(), step: None })
    };
    ($start:expr, $stop:expr) => {
        Slice::<isize>::from(Slice { start: $start.into(), stop: $stop.into(), step: None })
    };
    ($start:expr, $stop:expr, $step:expr) => {
        Slice::<isize>::from(Slice { start: $start.into(), stop: $stop.into(), step: $step.into() })
    };
}

#[macro_export]
macro_rules! s {
    // basic rule
    [$($slc:expr),*] => {
        &[$(($slc).into()),*]
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice() {
        let t = 3_usize;
        let s = slice!(1, 2, t);
        assert_eq!(s.start(), Some(1));
        assert_eq!(s.stop(), Some(2));
        assert_eq!(s.step(), Some(3));
    }

    #[test]
    fn test_slice_at_dim() {
        let l = Layout::<Ix3>::new([2, 3, 4], [1, 10, 100], 0);
        let s = slice!(10, 1, -1);
        let l1 = l.dim_narrow(1, s).unwrap();
        println!("{:?}", l1);
        let l2 = l.dim_select(1, -2).unwrap();
        println!("{:?}", l2);
        let l3 = l.dim_insert(1).unwrap();
        println!("{:?}", l3);

        let l = Layout::<Ix3>::new([2, 3, 4], [100, 10, 1], 0);
        let l3 = l.dim_insert(1).unwrap();
        println!("{:?}", l3);

        let l4 = l.dim_slice(s![Indexer::Ellipsis, 1..3, None, 2]).unwrap();
        let l4 = l4.into_dim::<Ix3>().unwrap();
        println!("{:?}", l4);
    }
}
