use crate::prelude_dev::*;

// type alias for this file
type Order = TensorIterOrder;

/* #region translate tensor order to col-major with TensorIterType */

/// This function will return a f-prefer layout that make minimal memory
/// accessing efforts (pointers will not frequently back-and-forth).
///
/// Note that this function should only be used for iteration.
///
/// # Parameters
///
/// - `keep_shape`: Keep size of output layout when input layout is boardcasted.
///   This option should be false if [`TensorIterOrder::K`] and true if
///   [`TensorIterOrder::G`].
///
/// # Returns
///
/// - `layout`: The output layout of greedy iteration.
/// - `index`: Transpose index from input layout to output layout.
pub fn greedy_layout<D>(layout: &Layout<D>, keep_shape: bool) -> (Layout<D>, Vec<usize>)
where
    D: DimDevAPI,
{
    // if no elements in layout, return itself
    if layout.size() == 0 {
        return (layout.clone(), (0..layout.ndim()).collect::<Vec<usize>>());
    }

    // revert negative strides
    let mut layout = layout.clone();
    for n in 0..layout.ndim() {
        if layout.stride()[n] < 0 {
            // should not panic here
            layout = layout.dim_narrow(n as isize, slice!(None, None, -1)).unwrap();
        }
    }

    let shape_old = layout.shape.as_ref();
    let stride_old = layout.stride.as_ref();

    let mut index = (0..layout.ndim()).collect::<Vec<usize>>();
    if keep_shape {
        // sort shape and strides if keep shape
        // - (shape = 1 / stride = 0) the smallest (pointer not moving for these cases)
        // - if (shape = 1 / stride = 0, broadcastable axes) then compare shape
        // - (larger shape first) if not broadcastable axes, then compare stride size
        //   (smaller stride first)
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1];
            let d2 = shape_old[i2];
            let t1 = stride_old[i1];
            let t2 = stride_old[i2];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => d2.cmp(&d1),
                (true, false) => core::cmp::Ordering::Less,
                (false, true) => core::cmp::Ordering::Greater,
                (false, false) => t1.cmp(&t2),
            }
        });
    } else {
        // sort shape and strides if not keep shape
        // everything is similar, though broadcastable axes should be moved to last
        index.sort_by(|&i1, &i2| {
            let d1 = shape_old[i1];
            let d2 = shape_old[i2];
            let t1 = stride_old[i1];
            let t2 = stride_old[i2];
            match (d1 == 1 || t1 == 0, d2 == 1 || t2 == 0) {
                (true, true) => d2.cmp(&d1),
                (true, false) => core::cmp::Ordering::Greater,
                (false, true) => core::cmp::Ordering::Less,
                (false, false) => t1.cmp(&t2),
            }
        });
    }

    let index_isize = index.iter().map(|&i| i as isize).collect::<Vec<isize>>();
    let mut layout = layout.transpose(&index_isize).unwrap();

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
pub fn translate_to_col_major_one<D>(
    layout: &Layout<D>,
    it_ord: TensorIterOrder,
) -> Result<Layout<D>>
where
    D: DimAPI,
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
    match it_ord {
        Order::C => fn_c(layout),
        Order::F => fn_f(layout),
        Order::A => {
            let c_contig = layout.is_c_contig();
            let f_contig = layout.is_f_contig();
            if c_contig || f_contig {
                fn_b(layout)
            } else {
                let c_prefer = layout.is_c_prefer();
                let f_prefer = layout.is_f_prefer();
                match (c_prefer, f_prefer) {
                    (true, false) => fn_c(layout),
                    (false, true) => fn_f(layout),
                    (_, _) => match TensorOrder::default() {
                        TensorOrder::C => fn_c(layout),
                        TensorOrder::F => fn_f(layout),
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
/// - B:sequential memory; valid option if `size = bound_max - bound_min`,
///   otherwise raise err
pub fn translate_to_col_major<D>(
    layouts: &[Layout<D>],
    it_ord: TensorIterOrder,
) -> Result<Vec<Layout<D>>>
where
    D: DimAPI,
{
    // this function will map all layouts to column-major iteration by a single
    // iter-order.
    let fn_single = |ls: &[Layout<D>], it_type| {
        ls.iter().map(|l| translate_to_col_major_one(l, it_type)).collect()
    };

    // make sure all layouts have the same shape
    let is_same_shape = layouts.windows(2).all(|w| w[0].shape() == w[1].shape());
    rstsr_assert!(is_same_shape, InvalidLayout, "All layouts in this function must be the same.")?;

    match it_ord {
        Order::C | Order::F | Order::B => fn_single(layouts, it_ord),
        Order::A => {
            let c_contig = layouts.iter().all(Layout::is_c_contig);
            let f_contig = layouts.iter().all(Layout::is_f_contig);
            if c_contig || f_contig {
                fn_single(layouts, TensorIterOrder::B)
            } else {
                let c_prefer = layouts.iter().all(Layout::is_c_prefer);
                let f_prefer = layouts.iter().all(Layout::is_f_prefer);
                match (c_prefer, f_prefer) {
                    (true, false) => fn_single(layouts, TensorIterOrder::C),
                    (false, true) => fn_single(layouts, TensorIterOrder::F),
                    (_, _) => match TensorOrder::default() {
                        TensorOrder::C => fn_single(layouts, TensorIterOrder::C),
                        TensorOrder::F => fn_single(layouts, TensorIterOrder::F),
                    },
                }
            }
        },
        Order::K => {
            // find the layout with the largest non-broadcast-size
            let size_iter = layouts.iter().map(|l| l.size_non_broadcast()).collect::<Vec<_>>();
            let idx_layout = if size_iter.iter().max() == size_iter.iter().min() {
                0
            } else {
                size_iter.into_iter().enumerate().max_by_key(|(_, v)| *v).unwrap_or((0, 0)).0
            };
            // make same permutation for all layouts
            let (_, permute_index) = greedy_layout(&layouts[idx_layout], true);
            let permute_index = permute_index.iter().map(|&i| i as isize).collect::<Vec<isize>>();
            layouts.iter().map(|l| l.transpose(&permute_index)).collect()
        },
        Order::G => rstsr_invalid!(it_ord, "This option is not valid for multiple layouts")?,
    }
}

/* #endregion */

/* #region new code */

/// Layout iterator.
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
pub struct LayoutColMajorIterator<D>
where
    D: DimDevAPI,
{
    layout: Layout<D>,

    index_start: D, // this is not used for buffer-order
    iter_start: usize,
    offset_start: usize,

    index_end: D, // this is not used for buffer-order
    iter_end: usize,
    offset_end: usize,

    iter_type: TensorIterType,
    is_order_buffer: bool,
}

impl<D> LayoutColMajorIterator<D>
where
    D: DimDevAPI,
{
    /// This function generates col-major (f-prefer) layout, then give its
    /// iterator object.
    pub fn new(layout: &Layout<D>, _it_type: Option<TensorIterType>) -> Result<Self> {
        let layout = layout.clone();
        let iter_start = 0;
        let iter_end = layout.size();
        let index_start = layout.new_shape();
        let index_end = layout.new_shape();
        let offset_start = layout.offset();
        let offset_end = unsafe { layout.index_uncheck(index_end.as_ref()) };
        let is_order_buffer =
            { layout.ndim() == 0 || layout.size() == layout.shape()[0] && layout.stride()[0] == 1 };

        return Ok(Self {
            layout,
            index_start,
            iter_start,
            offset_start,
            index_end,
            iter_end,
            offset_end,
            iter_type: _it_type.unwrap_or_default(),
            is_order_buffer,
        });
        // todo: implement it_type
    }
}

impl<D> LayoutColMajorIterator<D>
where
    D: DimDevAPI,
{
    #[inline]
    fn next_iter_index(&mut self) {
        // buffer-order fast return
        if self.is_order_buffer {
            self.iter_start += 1;
            self.offset_start += 1;
            return;
        }
        let layout = &self.layout;
        let index = self.index_start.as_mut();
        let mut offset = self.offset_start as isize;
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
                for (d, t, idx) in izip!(shape, stride, index.as_mut(),) {
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
        self.offset_start = offset as usize;
        self.iter_start += 1;
    }
}

impl<D> Iterator for LayoutColMajorIterator<D>
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
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for LayoutColMajorIterator<D>
where
    D: DimDevAPI,
{
    fn len(&self) -> usize {
        self.iter_end - self.iter_start
    }
}

#[test]
fn playground() {
    use std::time::Instant;
    let n = 1024 * 1024;
    let layout = [1024, 1024].f();
    let it = LayoutColMajorIterator::new(&layout, None).unwrap();
    let src = (0..n).map(|v| f64::from(v as u32)).collect::<Vec<f64>>();
    let mut val = 0.0;

    let now = Instant::now();
    for _ in 0..1000 {
        val = 0.0;
        for i in it.clone() {
            val += src[i];
        }
    }
    println!("Time: {:?}", now.elapsed());
    println!("{:}", val);

    let now = Instant::now();
    for _ in 0..1000 {
        val = 0.0;
        for i in it.clone() {
            val += src[i];
        }
    }
    println!("Time: {:?}", now.elapsed());
    println!("{:}", val);

    let now = Instant::now();
    for _ in 0..1000 {
        val = src.iter().sum()
    }
    println!("Time: {:?}", now.elapsed());
    println!("{:}", val);
}

/* #endregion */

/* #region old code */

/// Basic layout iteration trait. Any layout iteration struct should implement
/// this trait.
pub trait DimIterLayoutBaseAPI<It>: DimDevAPI {
    /// Iterator constructor
    fn new_it(layout: &Layout<Self>) -> Result<It>;
}

/// Trait for layout iteration, generates next index from previous for row-major
/// case.
pub trait DimIterLayoutAPI<It>: DimIterLayoutBaseAPI<It> {
    /// Get the next index, but note that this operation shall handle index
    /// iterator in-place.
    fn next_iter_index(it_obj: &mut It);
}

pub trait IterLayoutBaseAPI<D>: Sized
where
    D: DimIterLayoutBaseAPI<Self> + DimIterLayoutAPI<Self>,
{
    fn new_it(layout: &Layout<D>) -> Result<Self> {
        D::new_it(layout)
    }
    fn next_index(&mut self) {
        D::next_iter_index(self);
    }
}

/* #region row-major */

/// Basic layout iteration struct.
///
/// This iteration will naively iterate over all elements by row-major.
#[derive(Clone, Debug)]
pub struct IterLayoutC<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutC<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutC<D>> {
        type It<D> = IterLayoutC<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, offset });
        }
        let mut last_index = layout.shape().clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(It::<D> { layout, index: Some(last_index), offset });
    }
}

impl<const N: usize> DimIterLayoutAPI<IterLayoutC<Ix<N>>> for Ix<N> {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutC<Ix<N>>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        };
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match N {
            0 => {
                *index = None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                }
            },
            2 => {
                index_in[1] += 1;
                *offset = (*offset as isize + stride[1]) as usize;
                if index_in[1] == shape[1] {
                    index_in[1] = 0;
                    *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                    index_in[0] += 1;
                    *offset = (*offset as isize + stride[0]) as usize;
                    if index_in[0] == shape[0] {
                        *index = None;
                    }
                }
            },
            3 => {
                index_in[2] += 1;
                *offset = (*offset as isize + stride[2]) as usize;
                if index_in[2] == shape[2] {
                    index_in[2] = 0;
                    *offset = (*offset as isize - shape[2] as isize * stride[2]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        index_in[1] = 0;
                        *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                        index_in[0] += 1;
                        *offset = (*offset as isize + stride[0]) as usize;
                        if index_in[0] == shape[0] {
                            *index = None;
                        }
                    }
                }
            },
            4 => {
                index_in[3] += 1;
                *offset = (*offset as isize + stride[3]) as usize;
                if index_in[3] == shape[3] {
                    index_in[3] = 0;
                    *offset = (*offset as isize - shape[3] as isize * stride[3]) as usize;
                    index_in[2] += 1;
                    *offset = (*offset as isize + stride[2]) as usize;
                    if index_in[2] == shape[2] {
                        index_in[2] = 0;
                        *offset = (*offset as isize - shape[2] as isize * stride[2]) as usize;
                        index_in[1] += 1;
                        *offset = (*offset as isize + stride[1]) as usize;
                        if index_in[1] == shape[1] {
                            index_in[1] = 0;
                            *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                            index_in[0] += 1;
                            *offset = (*offset as isize + stride[0]) as usize;
                            if index_in[0] == shape[0] {
                                *index = None;
                            }
                        }
                    }
                }
            },
            _ => {
                let mut done = false;
                for (d, t, idx) in izip!(shape, stride, index_in).rev() {
                    *idx += 1;
                    *offset = (*offset as isize + t) as usize;
                    if idx == d {
                        *idx = 0;
                        *offset = (*offset as isize - *d as isize * t) as usize;
                    } else {
                        done = true;
                        break;
                    }
                }
                if !done {
                    *index = None;
                }
            },
        }
    }
}

impl DimIterLayoutAPI<IterLayoutC<IxD>> for IxD {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutC<IxD>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape().as_ref();
        let stride: &[isize] = layout.stride().as_ref();
        let mut done = false;
        for (d, t, idx) in izip!(shape, stride, index_in).rev() {
            *idx += 1;
            *offset = (*offset as isize + t) as usize;
            if idx == d {
                *idx = 0;
                *offset = (*offset as isize - *d as isize * t) as usize;
            } else {
                done = true;
                break;
            }
        }
        if !done {
            *index = None;
        }
    }
}

impl<D> Iterator for IterLayoutC<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutC<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutC<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_row_major() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(4, 3, 2).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].f();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutC::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22, 1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23]
    );
}

/* #endregion */

/* #region col-major */

/// Basic layout iteration struct.
///
/// This iteration will naively iterate over all elements by row-major.
#[derive(Clone, Debug)]
pub struct IterLayoutF<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<D>,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutF<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutF<D>> {
        type It<D> = IterLayoutF<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, offset });
        }
        let mut last_index = layout.shape().clone();
        for i in 0..layout.ndim() {
            last_index[i] = 0;
        }
        return Ok(It::<D> { layout, index: Some(last_index), offset });
    }
}

impl<const N: usize> DimIterLayoutAPI<IterLayoutF<Ix<N>>> for Ix<N> {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutF<Ix<N>>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape = layout.shape().as_ref();
        let stride = layout.stride().as_ref();
        match N {
            0 => {
                *index = None;
            },
            1 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    *index = None;
                }
            },
            2 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        *index = None;
                    }
                }
            },
            3 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        index_in[1] = 0;
                        *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                        index_in[2] += 1;
                        *offset = (*offset as isize + stride[2]) as usize;
                        if index_in[2] == shape[2] {
                            *index = None;
                        }
                    }
                }
            },
            4 => {
                index_in[0] += 1;
                *offset = (*offset as isize + stride[0]) as usize;
                if index_in[0] == shape[0] {
                    index_in[0] = 0;
                    *offset = (*offset as isize - shape[0] as isize * stride[0]) as usize;
                    index_in[1] += 1;
                    *offset = (*offset as isize + stride[1]) as usize;
                    if index_in[1] == shape[1] {
                        index_in[1] = 0;
                        *offset = (*offset as isize - shape[1] as isize * stride[1]) as usize;
                        index_in[2] += 1;
                        *offset = (*offset as isize + stride[2]) as usize;
                        if index_in[2] == shape[2] {
                            index_in[2] = 0;
                            *offset = (*offset as isize - shape[2] as isize * stride[2]) as usize;
                            index_in[3] += 1;
                            *offset = (*offset as isize + stride[3]) as usize;
                            if index_in[3] == shape[3] {
                                *index = None;
                            }
                        }
                    }
                }
            },
            _ => {
                let mut done = false;
                for (d, t, idx) in izip!(shape, stride, index_in.as_mut(),) {
                    *idx += 1;
                    *offset = (*offset as isize + t) as usize;
                    if idx == d {
                        *idx = 0;
                        *offset = (*offset as isize - *d as isize * t) as usize;
                    } else {
                        done = true;
                        break;
                    }
                }
                if !done {
                    *index = None;
                }
            },
        }
    }
}

impl DimIterLayoutAPI<IterLayoutF<IxD>> for IxD {
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutF<IxD>) {
        let (layout, index, offset) = (&it_obj.layout, &mut it_obj.index, &mut it_obj.offset);
        if index.is_none() {
            return;
        }
        let index_in = index.as_mut().unwrap();
        let shape: &[usize] = layout.shape().as_ref();
        let stride: &[isize] = layout.stride().as_ref();
        let mut done = false;
        for (d, t, idx) in izip!(shape, stride, index_in) {
            *idx += 1;
            *offset = (*offset as isize + t) as usize;
            if idx == d {
                *idx = 0;
                *offset = (*offset as isize - *d as isize * t) as usize;
            } else {
                done = true;
                break;
            }
        }
        if !done {
            *index = None;
        }
    }
}

impl<D> Iterator for IterLayoutF<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutF<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutF<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_col_major() {
    let layout = [2, 3, 4].f();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    // np.arange(24).reshape(2, 3, 4).transpose(2, 1, 0).flatten()
    let layout = [2, 3, 4].c();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutF::new_it(&layout).unwrap();
    assert_eq!(
        iter.collect::<Vec<_>>(),
        [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
}

/* #endregion */

/* #region mem-non-strided */

/// Iterator that only applies to layout that has contiguous memory (not exactly
/// same to c-contig or f-contig).
#[derive(Clone, Debug)]
pub struct IterLayoutMemNonStrided<D>
where
    D: DimDevAPI,
{
    pub(crate) layout: Layout<D>,
    index: Option<usize>,
    idx_max: usize,
    offset: usize,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutMemNonStrided<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutMemNonStrided<D>> {
        type It<D> = IterLayoutMemNonStrided<D>;
        let layout = layout.clone();
        let offset = layout.offset();
        if layout.size() == 0 {
            return Ok(It::<D> { layout, index: None, idx_max: 0, offset });
        }
        let (idx_min, idx_max) = layout.bounds_index()?;
        rstsr_assert_eq!(idx_max - idx_min, layout.size(), InvalidLayout)?;
        return Ok(It::<D> { layout, index: Some(idx_min), idx_max, offset });
    }
}

impl<D> DimIterLayoutAPI<IterLayoutMemNonStrided<D>> for D
where
    D: DimDevAPI,
{
    #[inline]
    fn next_iter_index(it_obj: &mut IterLayoutMemNonStrided<D>) {
        if let Some(index) = it_obj.index.as_mut() {
            *index += 1;
            it_obj.offset += 1;
            if *index == it_obj.idx_max {
                it_obj.index = None;
            }
        }
    }
}

impl<D> Iterator for IterLayoutMemNonStrided<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.as_ref()?;
        let offset = self.offset;
        D::next_iter_index(self);
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutMemNonStrided<D>
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutMemNonStrided<D> where D: DimIterLayoutAPI<Self> {}

#[test]
fn test_iter_layout_mem_non_strided() {
    let layout = [2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].c();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = [2, 3, 4].f();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
    let layout = vec![2, 3, 4].f().swapaxes(1, 2).unwrap();
    let iter = IterLayoutMemNonStrided::new_it(&layout).unwrap();
    assert_eq!(iter.collect::<Vec<_>>(), (0..24).collect::<Vec<_>>());
}

/* #endregion */

/* #region greedy-major */

pub struct IterLayoutGreedy<D>
where
    D: DimDevAPI,
{
    pub(crate) inner: IterLayoutF<D>,
}

impl<D> DimIterLayoutBaseAPI<IterLayoutGreedy<D>> for D
where
    D: DimDevAPI,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutGreedy<D>> {
        let layout = layout.clone();
        let inner = D::new_it(&layout)?;
        return Ok(IterLayoutGreedy::<D> { inner });
    }
}

impl<D> DimIterLayoutAPI<IterLayoutGreedy<D>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn next_iter_index(it_obj: &mut IterLayoutGreedy<D>) {
        D::next_iter_index(&mut it_obj.inner);
    }
}

impl<D> Iterator for IterLayoutGreedy<D>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.index.as_ref()?;
        let offset = self.inner.offset;
        self.inner.next_index();
        return Some(offset);
    }
}

impl<D> ExactSizeIterator for IterLayoutGreedy<D>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn len(&self) -> usize {
        self.inner.layout.size()
    }
}

impl<D> IterLayoutBaseAPI<D> for IterLayoutGreedy<D> where D: DimIterLayoutAPI<Self> {}

/* #endregion */

/* #region enum of iterator */

pub enum IterLayoutEnum<D, const CHG: bool>
where
    D: DimDevAPI,
{
    C(IterLayoutC<D>),
    F(IterLayoutF<D>),
    MemNonStrided(IterLayoutMemNonStrided<D>),
    GreedyMajor(IterLayoutGreedy<D>),
}

impl<D, const CHG: bool> DimIterLayoutBaseAPI<IterLayoutEnum<D, CHG>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn new_it(layout: &Layout<D>) -> Result<IterLayoutEnum<D, CHG>> {
        type It<D, const CHG: bool> = IterLayoutEnum<D, CHG>;
        // this implementation generates the most efficient iterator, but not the
        // standard layout.
        let layout = layout.clone();
        match CHG {
            false => match (layout.is_c_prefer(), layout.is_f_prefer()) {
                (true, false) => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                (false, true) => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                (_, _) => match TensorOrder::default() {
                    TensorOrder::C => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                    TensorOrder::F => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                },
            },
            true => {
                let iter_mem_non_strided = IterLayoutMemNonStrided::new_it(&layout);
                if let Ok(it) = iter_mem_non_strided {
                    Ok(It::<D, CHG>::MemNonStrided(it))
                } else {
                    match (layout.is_c_prefer(), layout.is_f_prefer()) {
                        (true, false) => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                        (false, true) => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                        (true, true) => match TensorOrder::default() {
                            TensorOrder::C => Ok(It::<D, CHG>::C(IterLayoutC::new_it(&layout)?)),
                            TensorOrder::F => Ok(It::<D, CHG>::F(IterLayoutF::new_it(&layout)?)),
                        },
                        (false, false) => {
                            Ok(It::<D, CHG>::GreedyMajor(IterLayoutGreedy::new_it(&layout)?))
                        },
                    }
                }
            },
        }
    }
}

impl<D, const CHG: bool> DimIterLayoutAPI<IterLayoutEnum<D, CHG>> for D
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn next_iter_index(it_obj: &mut IterLayoutEnum<D, CHG>) {
        type It<D, const CHG: bool> = IterLayoutEnum<D, CHG>;
        match it_obj {
            It::<D, CHG>::C(it) => it.next_index(),
            It::<D, CHG>::F(it) => it.next_index(),
            It::<D, CHG>::MemNonStrided(it) => it.next_index(),
            It::<D, CHG>::GreedyMajor(it) => it.next_index(),
        }
    }
}

impl<D, const CHG: bool> Iterator for IterLayoutEnum<D, CHG>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::C(it) => it.next(),
            Self::F(it) => it.next(),
            Self::MemNonStrided(it) => it.next(),
            Self::GreedyMajor(it) => it.next(),
        }
    }
}

impl<D, const CHG: bool> ExactSizeIterator for IterLayoutEnum<D, CHG>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    fn len(&self) -> usize {
        match self {
            Self::C(it) => it.len(),
            Self::F(it) => it.len(),
            Self::MemNonStrided(it) => it.len(),
            Self::GreedyMajor(it) => it.len(),
        }
    }
}

impl<D, const CHG: bool> IterLayoutBaseAPI<D> for IterLayoutEnum<D, CHG> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterLayoutType {
    C,
    F,
    MemNonStrided,
    GreedyMajor,
}

pub fn iter_layout_by_type<D>(
    ty: IterLayoutType,
    layout: &Layout<D>,
) -> Result<IterLayoutEnum<D, true>>
where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>,
{
    match ty {
        IterLayoutType::C => Ok(IterLayoutEnum::C(IterLayoutC::new_it(layout)?)),
        IterLayoutType::F => Ok(IterLayoutEnum::F(IterLayoutF::new_it(layout)?)),
        IterLayoutType::MemNonStrided => {
            Ok(IterLayoutEnum::MemNonStrided(IterLayoutMemNonStrided::new_it(layout)?))
        },
        IterLayoutType::GreedyMajor => {
            Ok(IterLayoutEnum::GreedyMajor(IterLayoutGreedy::new_it(layout)?))
        },
    }
}

/* #endregion */

pub trait IterLayoutAPI<D>:
    IterLayoutBaseAPI<D> + Iterator<Item = usize> + ExactSizeIterator
where
    D: DimDevAPI + DimIterLayoutAPI<Self>,
{
}

impl<D> IterLayoutAPI<D> for IterLayoutC<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutF<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutMemNonStrided<D> where D: DimDevAPI + DimIterLayoutAPI<Self> {}
impl<D> IterLayoutAPI<D> for IterLayoutGreedy<D> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutF<D>>
{
}
impl<D, const CHG: bool> IterLayoutAPI<D> for IterLayoutEnum<D, CHG> where
    D: DimDevAPI + DimIterLayoutAPI<IterLayoutC<D>> + DimIterLayoutAPI<IterLayoutF<D>>
{
}

/* #endregion */

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
            let expect = Layout::new_unchecked([3, 1, 1, 2, 6, 5], [0, 10, 40, 10, 100, 1000], 0);
            assert_eq!(greedy, expect);
        }
    }
}
