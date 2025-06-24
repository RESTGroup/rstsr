//! Layout of tensor.
use crate::prelude_dev::*;
use itertools::izip;

/* #region Struct Definitions */

/// Layout of tensor.
///
/// Layout is a struct that contains shape, stride, and offset of tensor.
/// - Shape is the size of each dimension of tensor.
/// - Stride is the number of elements to skip to get to the next element in each dimension.
/// - Offset is the starting position of tensor.
#[doc = include_str!("readme.md")]
#[derive(Clone)]
pub struct Layout<D>
where
    D: DimBaseAPI,
{
    // essential definitions to layout
    pub(crate) shape: D,
    pub(crate) stride: D::Stride,
    pub(crate) offset: usize,
}

unsafe impl<D> Send for Layout<D> where D: DimBaseAPI {}
unsafe impl<D> Sync for Layout<D> where D: DimBaseAPI {}

/* #endregion */

/* #region Layout */

/// Getter/setter functions for layout.
impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Shape of tensor. Getter function.
    #[inline]
    pub fn shape(&self) -> &D {
        &self.shape
    }

    /// Stride of tensor. Getter function.
    #[inline]
    pub fn stride(&self) -> &D::Stride {
        &self.stride
    }

    /// Starting offset of tensor. Getter function.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of dimensions of tensor.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Total number of elements in tensor.
    ///
    /// # Note
    ///
    /// This function uses cached size, instead of evaluating from shape.
    #[inline]
    pub fn size(&self) -> usize {
        self.shape().as_ref().iter().product()
    }

    /// Manually set offset.
    ///
    /// # Safety
    ///
    /// We will not check whether this offset is valid or not.
    /// In most cases, it is not intended to be used by user.
    pub unsafe fn set_offset(&mut self, offset: usize) -> &mut Self {
        self.offset = offset;
        return self;
    }
}

/// Properties of layout.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Whether this tensor is f-preferred.
    pub fn f_prefer(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut last = 0;
        for (&s, &d) in stride.iter().zip(shape.iter()) {
            if d != 1 {
                if s < last {
                    // latter strides must larger than previous strides
                    return false;
                }
                if last == 0 && s != 1 {
                    // first stride must be 1
                    return false;
                }
                last = s;
            } else if last == 0 {
                // if dimension is one, then consider that stride is one, counted as contiguous
                // in last dimension
                last = 1;
            }
        }
        return true;
    }

    /// Whether this tensor is c-preferred.
    pub fn c_prefer(&self) -> bool {
        // always true for 0-dimension or 0-size tensor
        if self.ndim() == 0 || self.size() == 0 {
            return true;
        }

        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut last = 0;
        for (&s, &d) in stride.iter().zip(shape.iter()).rev() {
            if d != 1 {
                if s < last {
                    // previous strides must larger than latter strides
                    return false;
                }
                if last == 0 && s != 1 {
                    // last stride must be 1
                    return false;
                }
                last = s;
            } else if last == 0 {
                // if dimension is one, then consider that stride is one, counted as contiguous
                // in last dimension
                last = 1;
            }
        }
        return true;
    }

    /// Least number of dimensions that is f-contiguous for layout.
    ///
    /// This function can be useful determining when to iterate by contiguous,
    /// and when to iterate by index.
    pub fn ndim_of_f_contig(&self) -> usize {
        if self.ndim() == 0 || self.size() == 0 {
            return self.ndim();
        }
        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        for (ndim, (&s, &d)) in stride.iter().zip(shape.iter()).enumerate() {
            if d != 1 && s != acc {
                return ndim;
            }
            acc *= d as isize;
        }
        return self.ndim();
    }

    /// Least number of dimensions that is c-contiguous for layout.
    ///
    /// This function can be useful determining when to iterate by contiguous,
    /// and when to iterate by index.
    pub fn ndim_of_c_contig(&self) -> usize {
        if self.ndim() == 0 || self.size() == 0 {
            return self.ndim();
        }
        let stride = self.stride.as_ref();
        let shape = self.shape.as_ref();
        let mut acc = 1;
        for (ndim, (&s, &d)) in stride.iter().zip(shape.iter()).rev().enumerate() {
            if d != 1 && s != acc {
                return ndim;
            }
            acc *= d as isize;
        }
        return self.ndim();
    }

    /// Whether this tensor is f-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is not important.
    /// - When length of a dimension is zero, then tensor contains no elements, thus f-contiguous.
    pub fn f_contig(&self) -> bool {
        self.ndim() == self.ndim_of_f_contig()
    }

    /// Whether this tensor is c-contiguous.
    ///
    /// Special cases
    /// - When length of a dimension is one, then stride to that dimension is not important.
    /// - When length of a dimension is zero, then tensor contains no elements, thus c-contiguous.
    pub fn c_contig(&self) -> bool {
        self.ndim() == self.ndim_of_c_contig()
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance.
    pub fn index_f(&self, index: &[isize]) -> Result<usize> {
        rstsr_assert_eq!(index.len(), self.ndim(), InvalidLayout)?;
        let mut pos = self.offset() as isize;
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();

        for (&idx, &shp, &strd) in izip!(index.iter(), shape.iter(), stride.iter()) {
            let idx = if idx < 0 { idx + shp as isize } else { idx };
            rstsr_pattern!(idx, 0..(shp as isize), ValueOutOfRange)?;
            pos += strd * idx;
        }
        rstsr_pattern!(pos, 0.., ValueOutOfRange)?;
        return Ok(pos as usize);
    }

    /// Index of tensor by list of indexes to dimensions.
    ///
    /// This function does not optimized for performance. Negative index
    /// allowed.
    ///
    /// # Panics
    ///
    /// - Index greater than shape
    pub fn index(&self, index: &[isize]) -> usize {
        self.index_f(index).unwrap()
    }

    /// Index range bounds of current layout. This bound is [min, max), which
    /// could be feed into range (min..max). If min == max, then this layout
    /// should not contains any element.
    ///
    /// This function will raise error when minimum index is smaller than zero.
    pub fn bounds_index(&self) -> Result<(usize, usize)> {
        let n = self.ndim();
        let offset = self.offset;
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();

        if n == 0 {
            return Ok((offset, offset + 1));
        }

        let mut min = offset as isize;
        let mut max = offset as isize;

        for i in 0..n {
            if shape[i] == 0 {
                return Ok((offset, offset));
            }
            if stride[i] > 0 {
                max += stride[i] * (shape[i] as isize - 1);
            } else {
                min += stride[i] * (shape[i] as isize - 1);
            }
        }
        rstsr_pattern!(min, 0.., ValueOutOfRange)?;
        return Ok((min as usize, max as usize + 1));
    }

    /// Check if strides is correct (no elemenets can overlap).
    ///
    /// This will check if all number of elements in dimension of small strides
    /// is less than larger strides. For example of valid stride:
    /// ```output
    /// shape:  (3,    2,  6)  -> sorted ->  ( 3,   6,   2)
    /// stride: (3, -300, 15)  -> sorted ->  ( 3,  15, 300)
    /// number of elements:                    9,  90,
    /// stride of next dimension              15, 300,
    /// number of elem < stride of next dim?   +,   +,
    /// ```
    ///
    /// Special cases
    /// - if length of tensor is zero, then strides will always be correct.
    /// - if certain dimension is one, then check for this stride will be ignored.
    ///
    /// # TODO
    ///
    /// Correctness of this function is not fully ensured.
    pub fn check_strides(&self) -> Result<()> {
        let shape = self.shape.as_ref();
        let stride = self.stride.as_ref();
        rstsr_assert_eq!(shape.len(), stride.len(), InvalidLayout)?;
        let n = shape.len();

        // unconditionally ok if no elements (length of tensor is zero)
        // unconditionally ok if 0-dimension
        if self.size() == 0 || n == 0 {
            return Ok(());
        }

        let mut indices = (0..n).filter(|&k| shape[k] > 1).collect::<Vec<_>>();
        indices.sort_by_key(|&k| stride[k].abs());
        let shape_sorted = indices.iter().map(|&k| shape[k]).collect::<Vec<_>>();
        let stride_sorted = indices.iter().map(|&k| stride[k].unsigned_abs()).collect::<Vec<_>>();

        // elem_cum: cumulative number count of elements in tensor for small strides
        let mut elem_cum = 0;
        for i in 0..indices.len() {
            // following function also checks that stride could not be zero
            rstsr_pattern!(
                elem_cum,
                0..stride_sorted[i],
                InvalidLayout,
                "Either stride be zero, or stride too small that elements in tensor can be overlapped."
            )?;
            elem_cum += (shape_sorted[i] - 1) * stride_sorted[i];
        }
        return Ok(());
    }

    pub fn diagonal(
        &self,
        offset: Option<isize>,
        axis1: Option<isize>,
        axis2: Option<isize>,
    ) -> Result<Layout<<D as DimSmallerOneAPI>::SmallerOne>>
    where
        D: DimSmallerOneAPI,
    {
        // check if this layout is at least 2-dimension
        rstsr_assert!(self.ndim() >= 2, InvalidLayout)?;
        // unwrap optional parameters
        let offset = offset.unwrap_or(0);
        let axis1 = axis1.unwrap_or(0);
        let axis2 = axis2.unwrap_or(1);
        let axis1 = if axis1 < 0 { self.ndim() as isize + axis1 } else { axis1 };
        let axis2 = if axis2 < 0 { self.ndim() as isize + axis2 } else { axis2 };
        rstsr_pattern!(axis1, 0..self.ndim() as isize, ValueOutOfRange)?;
        rstsr_pattern!(axis2, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis1 = axis1 as usize;
        let axis2 = axis2 as usize;

        // shape and strides of last two dimensions
        let d1 = self.shape()[axis1] as isize;
        let d2 = self.shape()[axis2] as isize;
        let t1 = self.stride()[axis1];
        let t2 = self.stride()[axis2];

        // number of elements in diagonal, and starting offset
        let (offset_diag, d_diag) = if (-d2 + 1..0).contains(&offset) {
            let offset = -offset;
            let offset_diag = (self.offset() as isize + t1 * offset) as usize;
            let d_diag = (d1 - offset).min(d2) as usize;
            (offset_diag, d_diag)
        } else if (0..d1).contains(&offset) {
            let offset_diag = (self.offset() as isize + t2 * offset) as usize;
            let d_diag = (d2 - offset).min(d1) as usize;
            (offset_diag, d_diag)
        } else {
            (self.offset(), 0)
        };

        // build new layout
        let t_diag = t1 + t2;
        let mut shape_diag = vec![];
        let mut stride_diag = vec![];
        for i in 0..self.ndim() {
            if i != axis1 && i != axis2 {
                shape_diag.push(self.shape()[i]);
                stride_diag.push(self.stride()[i]);
            }
        }
        shape_diag.push(d_diag);
        stride_diag.push(t_diag);
        let layout_diag = Layout::new(shape_diag, stride_diag, offset_diag)?;
        return layout_diag.into_dim::<<D as DimSmallerOneAPI>::SmallerOne>();
    }
}

/// Constructors of layout. See also [`DimLayoutContigAPI`] layout from shape
/// directly.
impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Generate new layout by providing everything.
    ///
    /// # Error when
    ///
    /// - Shape and stride length mismatch
    /// - Strides is correct (no elements can overlap)
    /// - Minimum bound is not negative
    #[inline]
    pub fn new(shape: D, stride: D::Stride, offset: usize) -> Result<Self>
    where
        D: DimShapeAPI + DimStrideAPI,
    {
        let layout = unsafe { Layout::new_unchecked(shape, stride, offset) };
        layout.bounds_index()?;
        layout.check_strides()?;
        return Ok(layout);
    }

    /// Generate new layout by providing everything, without checking bounds and
    /// strides.
    ///
    /// # Safety
    ///
    /// This function does not check whether layout is valid.
    #[inline]
    pub unsafe fn new_unchecked(shape: D, stride: D::Stride, offset: usize) -> Self {
        Layout { shape, stride, offset }
    }

    /// New zero shape, which number of dimensions are the same to current
    /// layout.
    #[inline]
    pub fn new_shape(&self) -> D {
        self.shape.new_shape()
    }

    /// New zero stride, which number of dimensions are the same to current
    /// layout.
    #[inline]
    pub fn new_stride(&self) -> D::Stride {
        self.shape.new_stride()
    }
}

/// Manuplation of layout.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Transpose layout by permutation.
    ///
    /// # See also
    ///
    /// - [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    /// - [Python array API: `permute_dims`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html)
    pub fn transpose(&self, axes: &[isize]) -> Result<Self> {
        // check axes and cast to usize
        let n = self.ndim();
        rstsr_assert_eq!(
            axes.len(),
            n,
            InvalidLayout,
            "number of elements in axes should be the same to number of dimensions."
        )?;
        // no elements in axes can be the same
        let mut permut_used = vec![false; n];
        for &p in axes {
            let p = if p < 0 { p + n as isize } else { p };
            rstsr_pattern!(p, 0..n as isize, InvalidLayout)?;
            let p = p as usize;
            permut_used[p] = true;
        }
        rstsr_assert!(
            permut_used.iter().all(|&b| b),
            InvalidLayout,
            "axes should contain all elements from 0 to n-1."
        )?;
        let axes = axes.iter().map(|&p| if p < 0 { p + n as isize } else { p } as usize).collect::<Vec<_>>();

        let shape_old = self.shape();
        let stride_old = self.stride();
        let mut shape = self.new_shape();
        let mut stride = self.new_stride();
        for i in 0..self.ndim() {
            shape[i] = shape_old[axes[i]];
            stride[i] = stride_old[axes[i]];
        }
        return unsafe { Ok(Layout::new_unchecked(shape, stride, self.offset)) };
    }

    /// Transpose layout by permutation.
    ///
    /// This is the same function to [`Layout::transpose`]
    pub fn permute_dims(&self, axes: &[isize]) -> Result<Self> {
        self.transpose(axes)
    }

    /// Reverse axes of layout.
    pub fn reverse_axes(&self) -> Self {
        let shape_old = self.shape();
        let stride_old = self.stride();
        let mut shape = self.new_shape();
        let mut stride = self.new_stride();
        for i in 0..self.ndim() {
            shape[i] = shape_old[self.ndim() - i - 1];
            stride[i] = stride_old[self.ndim() - i - 1];
        }
        return unsafe { Layout::new_unchecked(shape, stride, self.offset) };
    }

    /// Swap axes of layout.
    pub fn swapaxes(&self, axis1: isize, axis2: isize) -> Result<Self> {
        let axis1 = if axis1 < 0 { self.ndim() as isize + axis1 } else { axis1 };
        rstsr_pattern!(axis1, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis1 = axis1 as usize;

        let axis2 = if axis2 < 0 { self.ndim() as isize + axis2 } else { axis2 };
        rstsr_pattern!(axis2, 0..self.ndim() as isize, ValueOutOfRange)?;
        let axis2 = axis2 as usize;

        let mut shape = self.shape().clone();
        let mut stride = self.stride().clone();
        shape.as_mut().swap(axis1, axis2);
        stride.as_mut().swap(axis1, axis2);
        return unsafe { Ok(Layout::new_unchecked(shape, stride, self.offset)) };
    }
}

/// Fast indexing and utilities of layout.
///
/// These functions are mostly internal to this crate.
impl<D> Layout<D>
where
    D: DimBaseAPI + DimShapeAPI + DimStrideAPI,
{
    /// Index of tensor by list of indexes to dimensions.
    ///
    /// # Safety
    ///
    /// This function does not check for bounds, including
    /// - Negative index
    /// - Index greater than shape
    ///
    /// Due to these reasons, this function may well give index smaller than
    /// zero, which may occur in iterator; so this function returns isize.
    #[inline]
    pub unsafe fn index_uncheck(&self, index: &[usize]) -> isize {
        let stride = self.stride.as_ref();
        match self.ndim() {
            0 => self.offset as isize,
            1 => self.offset as isize + stride[0] * index[0] as isize,
            2 => self.offset as isize + stride[0] * index[0] as isize + stride[1] * index[1] as isize,
            3 => {
                self.offset as isize
                    + stride[0] * index[0] as isize
                    + stride[1] * index[1] as isize
                    + stride[2] * index[2] as isize
            },
            4 => {
                self.offset as isize
                    + stride[0] * index[0] as isize
                    + stride[1] * index[1] as isize
                    + stride[2] * index[2] as isize
                    + stride[3] * index[3] as isize
            },
            _ => {
                let mut pos = self.offset as isize;
                stride.iter().zip(index.iter()).for_each(|(&s, &i)| pos += s * i as isize);
                pos
            },
        }
    }
}

impl<D> PartialEq for Layout<D>
where
    D: DimBaseAPI,
{
    /// For layout, shape must be the same, while stride should be the same when
    /// shape is not zero or one, but can be arbitary otherwise.
    fn eq(&self, other: &Self) -> bool {
        if self.ndim() != other.ndim() {
            return false;
        }
        for i in 0..self.ndim() {
            let s1 = self.shape()[i];
            let s2 = other.shape()[i];
            if s1 != s2 {
                return false;
            }
            if s1 != 1 && s1 != 0 && self.stride()[i] != other.stride()[i] {
                return false;
            }
        }
        return true;
    }
}

pub trait DimLayoutContigAPI: DimBaseAPI + DimShapeAPI + DimStrideAPI {
    /// Generate new layout by providing shape and offset; stride fits into
    /// c-contiguous.
    fn new_c_contig(&self, offset: Option<usize>) -> Layout<Self> {
        let shape = self.clone();
        let stride = shape.stride_c_contig();
        unsafe { Layout::new_unchecked(shape, stride, offset.unwrap_or(0)) }
    }

    /// Generate new layout by providing shape and offset; stride fits into
    /// f-contiguous.
    fn new_f_contig(&self, offset: Option<usize>) -> Layout<Self> {
        let shape = self.clone();
        let stride = shape.stride_f_contig();
        unsafe { Layout::new_unchecked(shape, stride, offset.unwrap_or(0)) }
    }

    /// Simplified function to generate c-contiguous layout. See also
    /// [DimLayoutContigAPI::new_c_contig].
    fn c(&self) -> Layout<Self> {
        self.new_c_contig(None)
    }

    /// Simplified function to generate f-contiguous layout. See also
    /// [DimLayoutContigAPI::new_f_contig].
    fn f(&self) -> Layout<Self> {
        self.new_f_contig(None)
    }

    /// Generate new layout by providing shape, offset and order.
    fn new_contig(&self, offset: Option<usize>, order: FlagOrder) -> Layout<Self> {
        match order {
            FlagOrder::C => self.new_c_contig(offset),
            FlagOrder::F => self.new_f_contig(offset),
        }
    }
}

impl<const N: usize> DimLayoutContigAPI for Ix<N> {}
impl DimLayoutContigAPI for IxD {}

/* #endregion Layout */

/* #region Dimension Conversion */

pub trait DimIntoAPI<D>: DimBaseAPI
where
    D: DimBaseAPI,
{
    fn into_dim(layout: Layout<Self>) -> Result<Layout<D>>;
}

impl<D> DimIntoAPI<D> for IxD
where
    D: DimBaseAPI,
{
    fn into_dim(layout: Layout<IxD>) -> Result<Layout<D>> {
        let shape = layout.shape().clone().try_into().map_err(|_| rstsr_error!(InvalidLayout))?;
        let stride = layout.stride().clone().try_into().map_err(|_| rstsr_error!(InvalidLayout))?;
        let offset = layout.offset();
        return Ok(Layout { shape, stride, offset });
    }
}

impl<const N: usize> DimIntoAPI<IxD> for Ix<N> {
    fn into_dim(layout: Layout<Ix<N>>) -> Result<Layout<IxD>> {
        let shape = (*layout.shape()).into();
        let stride = (*layout.stride()).into();
        let offset = layout.offset();
        return Ok(Layout { shape, stride, offset });
    }
}

impl<const N: usize, const M: usize> DimIntoAPI<Ix<M>> for Ix<N> {
    fn into_dim(layout: Layout<Ix<N>>) -> Result<Layout<Ix<M>>> {
        rstsr_assert_eq!(N, M, InvalidLayout)?;
        let shape = layout.shape().to_vec().try_into().unwrap();
        let stride = layout.stride().to_vec().try_into().unwrap();
        let offset = layout.offset();
        return Ok(Layout { shape, stride, offset });
    }
}

impl<D> Layout<D>
where
    D: DimBaseAPI,
{
    /// Convert layout to another dimension.
    pub fn into_dim<D2>(self) -> Result<Layout<D2>>
    where
        D2: DimBaseAPI,
        D: DimIntoAPI<D2>,
    {
        D::into_dim(self)
    }

    /// Convert layout to another dimension.
    pub fn to_dim<D2>(&self) -> Result<Layout<D2>>
    where
        D2: DimBaseAPI,
        D: DimIntoAPI<D2>,
    {
        D::into_dim(self.clone())
    }
}

impl<const N: usize> From<Ix<N>> for Layout<Ix<N>> {
    fn from(shape: Ix<N>) -> Self {
        let stride = shape.stride_contig();
        Layout { shape, stride, offset: 0 }
    }
}

impl From<IxD> for Layout<IxD> {
    fn from(shape: IxD) -> Self {
        let stride = shape.stride_contig();
        Layout { shape, stride, offset: 0 }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn test_layout_new() {
        // a successful layout new
        let shape = [3, 2, 6];
        let stride = [3, -300, 15];
        let layout = Layout::new(shape, stride, 917).unwrap();
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[3, -300, 15]);
        assert_eq!(layout.offset(), 917);
        assert_eq!(layout.ndim(), 3);
        // unsuccessful layout new (offset underflow)
        let shape = [3, 2, 6];
        let stride = [3, -300, 15];
        let layout = Layout::new(shape, stride, 0);
        assert!(layout.is_err());
        // unsuccessful layout new (zero stride for non-0/1 shape)
        let shape = [3, 2, 6];
        let stride = [3, -300, 0];
        let layout = Layout::new(shape, stride, 1000);
        assert!(layout.is_err());
        // unsuccessful layout new (stride too small)
        let shape = [3, 2, 6];
        let stride = [3, 4, 7];
        let layout = Layout::new(shape, stride, 1000);
        assert!(layout.is_err());
        // successful layout new (zero dim)
        let shape = [];
        let stride = [];
        let layout = Layout::new(shape, stride, 1000);
        assert!(layout.is_ok());
        // successful layout new (stride 0 for 1-shape)
        let shape = [3, 1, 5];
        let stride = [1, 0, 15];
        let layout = Layout::new(shape, stride, 1);
        assert!(layout.is_ok());
        // successful layout new (stride 0 for 1-shape)
        let shape = [3, 1, 5];
        let stride = [1, 0, 15];
        let layout = Layout::new(shape, stride, 1);
        assert!(layout.is_ok());
        // successful layout new (zero-size tensor)
        let shape = [3, 0, 5];
        let stride = [-1, -2, -3];
        let layout = Layout::new(shape, stride, 1);
        assert!(layout.is_ok());
        // anyway, if one need custom layout, use new_unchecked
        let shape = [3, 2, 6];
        let stride = [3, -300, 0];
        let r = catch_unwind(|| unsafe { Layout::new_unchecked(shape, stride, 1000) });
        assert!(r.is_ok());
    }

    #[test]
    fn test_is_f_prefer() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [1, 10, 100], 0).unwrap();
        assert!(layout.f_prefer());
        let layout = Layout::new(shape, [1, 3, 15], 0).unwrap();
        assert!(layout.f_prefer());
        let layout = Layout::new(shape, [1, 3, -15], 1000).unwrap();
        assert!(!layout.f_prefer());
        let layout = Layout::new(shape, [1, 21, 3], 0).unwrap();
        assert!(!layout.f_prefer());
        let layout = Layout::new(shape, [35, 7, 1], 0).unwrap();
        assert!(!layout.f_prefer());
        let layout = Layout::new(shape, [2, 6, 30], 0).unwrap();
        assert!(!layout.f_prefer());
        // zero dimension
        let layout = Layout::new([], [], 0).unwrap();
        assert!(layout.f_prefer());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0).unwrap();
        assert!(layout.f_prefer());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [1, 1, 2], 0).unwrap();
        assert!(layout.f_prefer());
    }

    #[test]
    fn test_is_c_prefer() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [100, 10, 1], 0).unwrap();
        assert!(layout.c_prefer());
        let layout = Layout::new(shape, [35, 7, 1], 0).unwrap();
        assert!(layout.c_prefer());
        let layout = Layout::new(shape, [-35, 7, 1], 1000).unwrap();
        assert!(!layout.c_prefer());
        let layout = Layout::new(shape, [7, 21, 1], 0).unwrap();
        assert!(!layout.c_prefer());
        let layout = Layout::new(shape, [1, 3, 15], 0).unwrap();
        assert!(!layout.c_prefer());
        let layout = Layout::new(shape, [70, 14, 2], 0).unwrap();
        assert!(!layout.c_prefer());
        // zero dimension
        let layout = Layout::new([], [], 0).unwrap();
        assert!(layout.c_prefer());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0).unwrap();
        assert!(layout.c_prefer());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [4, 1, 1], 0).unwrap();
        assert!(layout.c_prefer());
    }

    #[test]
    fn test_is_f_contig() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [1, 3, 15], 0).unwrap();
        assert!(layout.f_contig());
        let layout = Layout::new(shape, [1, 4, 20], 0).unwrap();
        assert!(!layout.f_contig());
        // zero dimension
        let layout = Layout::new([], [], 0).unwrap();
        assert!(layout.f_contig());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0).unwrap();
        assert!(layout.f_contig());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [1, 1, 2], 0).unwrap();
        assert!(layout.f_contig());
    }

    #[test]
    fn test_is_c_contig() {
        // general case
        let shape = [3, 5, 7];
        let layout = Layout::new(shape, [35, 7, 1], 0).unwrap();
        assert!(layout.c_contig());
        let layout = Layout::new(shape, [36, 7, 1], 0).unwrap();
        assert!(!layout.c_contig());
        // zero dimension
        let layout = Layout::new([], [], 0).unwrap();
        assert!(layout.c_contig());
        // zero size
        let layout = Layout::new([2, 0, 4], [1, 10, 100], 0).unwrap();
        assert!(layout.c_contig());
        // shape with 1
        let layout = Layout::new([2, 1, 4], [4, 1, 1], 0).unwrap();
        assert!(layout.c_contig());
    }

    #[test]
    fn test_index() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        assert_eq!(layout.index(&[0, 0, 0]), 782);
        assert_eq!(layout.index(&[2, 1, 4]), 668);
        assert_eq!(layout.index(&[1, -2, -3]), 830);
        // zero-dim
        let layout = Layout::new([], [], 10).unwrap();
        assert_eq!(layout.index(&[]), 10);
    }

    #[test]
    fn test_bounds_index() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        // a.min() = 602, a.max() = 863
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        assert_eq!(layout.bounds_index().unwrap(), (602, 864));
        // situation that fails
        let layout = unsafe { Layout::new_unchecked([3, 2, 6], [3, -180, 15], 15) };
        assert!(layout.bounds_index().is_err());
        // zero-dim
        let layout = Layout::new([], [], 10).unwrap();
        assert_eq!(layout.bounds_index().unwrap(), (10, 11));
    }

    #[test]
    fn test_transpose() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        let trans = layout.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // permute_dims is alias of transpose
        let trans = layout.permute_dims(&[2, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // negative axis also allowed
        let trans = layout.transpose(&[-1, 0, 1]).unwrap();
        assert_eq!(trans.shape(), &[6, 3, 2]);
        assert_eq!(trans.stride(), &[15, 3, -180]);
        // repeated axis
        let trans = layout.transpose(&[-2, 0, 1]);
        assert!(trans.is_err());
        // non-valid dimension
        let trans = layout.transpose(&[1, 0]);
        assert!(trans.is_err());
        // zero-dim
        let layout = Layout::new([], [], 0).unwrap();
        let trans = layout.transpose(&[]);
        assert!(trans.is_ok());
    }

    #[test]
    fn test_reverse_axes() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        let trans = layout.reverse_axes();
        assert_eq!(trans.shape(), &[6, 2, 3]);
        assert_eq!(trans.stride(), &[15, -180, 3]);
        // zero-dim
        let layout = Layout::new([], [], 782).unwrap();
        let trans = layout.reverse_axes();
        assert_eq!(trans.shape(), &[]);
        assert_eq!(trans.stride(), &[]);
    }

    #[test]
    fn test_swapaxes() {
        // general
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        let trans = layout.swapaxes(-1, -2).unwrap();
        assert_eq!(trans.shape(), &[3, 6, 2]);
        assert_eq!(trans.stride(), &[3, 15, -180]);
        // same index is allowed
        let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
        let trans = layout.swapaxes(-1, -1).unwrap();
        assert_eq!(trans.shape(), &[3, 2, 6]);
        assert_eq!(trans.stride(), &[3, -180, 15]);
    }

    #[test]
    fn test_index_uncheck() {
        // a = np.arange(9 * 12 * 15)
        //       .reshape(9, 12, 15)[4:2:-1, 4:10, 2:10:3]
        //       .transpose(2, 0, 1)
        unsafe {
            // fixed dim
            let layout = Layout::new([3, 2, 6], [3, -180, 15], 782).unwrap();
            assert_eq!(layout.index_uncheck(&[0, 0, 0]), 782);
            assert_eq!(layout.index_uncheck(&[2, 1, 4]), 668);
            // dynamic dim
            let layout = Layout::new(vec![3, 2, 6], vec![3, -180, 15], 782).unwrap();
            assert_eq!(layout.index_uncheck(&[0, 0, 0]), 782);
            assert_eq!(layout.index_uncheck(&[2, 1, 4]), 668);
            // zero-dim
            let layout = Layout::new([], [], 10).unwrap();
            assert_eq!(layout.index_uncheck(&[]), 10);
        }
    }

    #[test]
    fn test_diagonal() {
        let layout = [2, 3, 4].c();
        let diag = layout.diagonal(None, None, None).unwrap();
        assert_eq!(diag, Layout::new([4, 2], [1, 16], 0).unwrap());
        let diag = layout.diagonal(Some(-1), Some(-2), Some(-1)).unwrap();
        assert_eq!(diag, Layout::new([2, 2], [12, 5], 0).unwrap());
        let diag = layout.diagonal(Some(-4), Some(-2), Some(-1)).unwrap();
        assert_eq!(diag, Layout::new([2, 0], [12, 5], 0).unwrap());
    }

    #[test]
    fn test_new_contig() {
        let layout = [3, 2, 6].c();
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[12, 6, 1]);
        let layout = [3, 2, 6].f();
        assert_eq!(layout.shape(), &[3, 2, 6]);
        assert_eq!(layout.stride(), &[1, 3, 6]);
        // following code generates contiguous layout
        // c/f-contig depends on cargo feature
        let layout: Layout<_> = [3, 2, 6].into();
        println!("{layout:?}");
    }

    #[test]
    fn test_layout_cast() {
        let layout = [3, 2, 6].c();
        assert!(layout.clone().into_dim::<IxD>().is_ok());
        assert!(layout.clone().into_dim::<Ix3>().is_ok());
        let layout = vec![3, 2, 6].c();
        assert!(layout.clone().into_dim::<IxD>().is_ok());
        assert!(layout.clone().into_dim::<Ix3>().is_ok());
        assert!(layout.clone().into_dim::<Ix2>().is_err());
    }

    #[test]
    fn test_unravel_index() {
        unsafe {
            let shape = [3, 2, 6];
            assert_eq!(shape.unravel_index_f(0), [0, 0, 0]);
            assert_eq!(shape.unravel_index_f(16), [1, 1, 2]);
            assert_eq!(shape.unravel_index_c(0), [0, 0, 0]);
            assert_eq!(shape.unravel_index_c(16), [1, 0, 4]);
        }
    }

    #[test]
    fn fix_too_strict_stride_check() {
        let layout = [10, 11, 12].c();
        let slc = (.., slice!(-1, 0, -4));
        let slc: AxesIndex<Indexer> = slc.try_into().unwrap();
        let indexed = layout.dim_slice(slc.as_ref()).unwrap();
        assert_eq!(indexed.shape(), &[10, 3, 12]);
        assert_eq!(indexed.stride(), &[132, -48, 1]);
    }
}
