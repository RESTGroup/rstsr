use crate::prelude_dev::*;
use core::sync::atomic::{AtomicUsize, Ordering};

pub static MIN_PRINT: AtomicUsize = AtomicUsize::new(3);
pub static MAX_PRINT: AtomicUsize = AtomicUsize::new(8);

pub struct FnPrintVecWithLayout<'v, 'l, 'f1, 'f2, T, D>
where
    T: Clone,
    D: DimAPI,
{
    vec: &'v [T],
    layout: &'l Layout<D>,
    offset: usize,
    idx_prev: Vec<usize>,
    max_print: usize,
    min_print: usize,
    fmt: &'f1 mut core::fmt::Formatter<'f2>,
}

/* #region display print */

pub fn print_vec_with_layout_dfs<T, D>(c: &mut FnPrintVecWithLayout<T, D>) -> core::fmt::Result
where
    T: Clone + Display,
    D: DimAPI,
{
    let FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt } = c;
    let max_print = *max_print;
    let min_print = *min_print;
    let ndim = layout.ndim();
    let offset = *offset;
    let len_prev = idx_prev.len();
    let shape = layout.shape().as_ref();
    let stride = layout.stride().as_ref();

    // special case
    if ndim == 0 {
        return write!(fmt, "{}", vec[layout.offset()]);
    }

    if idx_prev.last().is_some_and(|&v| v == shape[len_prev - 1]) {
        // hit shape boundary

        // special case: zero shape quick return
        if idx_prev.last().is_some_and(|&v| v == 0) {
            return write!(fmt, "[]");
        }

        // pop last index, increase previous index by 1 or skip
        if len_prev == 1 {
            // multiple dimension exit
            return Ok(());
        } else {
            let p1_prev = idx_prev.pop().unwrap();
            let p2_prev = idx_prev.pop().unwrap();
            let offset = offset as isize
                - p1_prev as isize * stride[len_prev - 1]
                - p2_prev as isize * stride[len_prev - 2];
            if shape[len_prev - 2] < max_print
                || p2_prev + min_print >= shape[len_prev - 2]
                || p2_prev + 1 < min_print
            {
                let p2 = p2_prev + 1;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs(c);
            } else {
                write!(fmt, "{}...\n\n", " ".repeat(len_prev - 1))?;
                let p2: usize = shape[len_prev - 2] - min_print;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs(c);
            }
        }
    } else {
        // not hit shape boundary
        if len_prev + 1 != ndim {
            // new index can still be pushed
            idx_prev.push(0);
            return print_vec_with_layout_dfs(c);
        } else {
            // last line

            // number of last dimension
            let nlast = *shape.last().unwrap();
            let stride_last = *stride.last().unwrap();

            // special case: zero shape quick return
            if nlast == 0 {
                return write!(fmt, "[]");
            }

            // prefix: "  [[["
            // count zeros from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "[")?;
            } else {
                let mut nbra = 1;
                for &idx in idx_prev.iter().rev() {
                    if idx == 0 {
                        nbra += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", " ".repeat(ndim - nbra))?;
                write!(fmt, "{:}", "[".repeat(nbra))?;
            }

            // values: " 1.23 4.56 ... 7.89 10.11"
            if nlast <= max_print.max(2 * min_print + 1) {
                // all elements in last dimension should be printed
                for i in 0..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
            } else {
                // only print the first/last min_print elements
                for i in 0..min_print {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
                write!(fmt, " ...")?;
                for i in (nlast - min_print)..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Display::fmt(&vec[offset_i], fmt)?;
                }
            };

            // suffix: "]]]"
            // count (if index = shape) from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "]")?;
            } else {
                let mut nket = 1;
                for i in (0..idx_prev.len()).rev() {
                    if idx_prev[i] == shape[i] - 1 {
                        nket += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", "]".repeat(nket))?;
                if nket != ndim {
                    // last line should not add new-line character
                    if nket > 1 {
                        writeln!(fmt, "\n")?;
                    } else {
                        writeln!(fmt)?;
                    }
                }
            }

            // pop last index, increase previous index by 1 or skip
            if len_prev == 0 {
                // one-dimension exit
                return Ok(());
            } else {
                let p1_prev = idx_prev.pop().unwrap();
                let offset = offset as isize - p1_prev as isize * stride[len_prev - 1];
                if shape[len_prev - 1] < max_print
                    || p1_prev + min_print >= shape[len_prev - 1]
                    || p1_prev + 1 < min_print
                {
                    let p1 = p1_prev + 1;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs(c);
                } else {
                    writeln!(fmt, "{}...", " ".repeat(len_prev))?;
                    let p1: usize = shape[len_prev - 1] - min_print;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs(c);
                }
            }
        }
    }
}

pub fn print_vec_with_layout<T, D>(
    fmt: &mut core::fmt::Formatter<'_>,
    vec: &[T],
    layout: &Layout<D>,
    max_print: usize,
    min_print: usize,
) -> core::fmt::Result
where
    T: Clone + Display,
    D: crate::layout::DimAPI,
{
    let idx_prev = vec![];
    let offset = layout.offset;
    let mut config =
        FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt };
    print_vec_with_layout_dfs(&mut config)
}

impl<R, T, B, D> Display for TensorAny<R, T, B, D>
where
    T: Clone + Display,
    B: DeviceAPI<T>,
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let vec = self.storage().to_cpu_vec().unwrap();
        let layout = &self.layout();
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        print_vec_with_layout(f, &vec, layout, max_print, min_print)
    }
}

/* #endregion */

/* #region debug print */

pub fn print_vec_with_layout_dfs_debug<T, D>(
    c: &mut FnPrintVecWithLayout<T, D>,
) -> core::fmt::Result
where
    T: Clone + Debug,
    D: DimAPI,
{
    let FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt } = c;
    let max_print = *max_print;
    let min_print = *min_print;
    let ndim = layout.ndim();
    let offset = *offset;
    let len_prev = idx_prev.len();
    let shape = layout.shape().as_ref();
    let stride = layout.stride().as_ref();

    // special case
    if ndim == 0 {
        return write!(fmt, "{:?}", vec[layout.offset()]);
    }

    if idx_prev.last().is_some_and(|&v| v == shape[len_prev - 1]) {
        // hit shape boundary

        // special case: zero shape quick return
        if idx_prev.last().is_some_and(|&v| v == 0) {
            return write!(fmt, "[]");
        }

        // pop last index, increase previous index by 1 or skip
        if len_prev == 1 {
            // multiple dimension exit
            return Ok(());
        } else {
            let p1_prev = idx_prev.pop().unwrap();
            let p2_prev = idx_prev.pop().unwrap();
            let offset = offset as isize
                - p1_prev as isize * stride[len_prev - 1]
                - p2_prev as isize * stride[len_prev - 2];
            if shape[len_prev - 2] < max_print
                || p2_prev + min_print >= shape[len_prev - 2]
                || p2_prev + 1 < min_print
            {
                let p2 = p2_prev + 1;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs_debug(c);
            } else {
                write!(fmt, "{}...\n\n", " ".repeat(len_prev - 1))?;
                let p2: usize = shape[len_prev - 2] - min_print;
                idx_prev.push(p2);
                let offset = offset + p2 as isize * stride[len_prev - 2];
                let offset = offset as usize;
                c.offset = offset;
                return print_vec_with_layout_dfs_debug(c);
            }
        }
    } else {
        // not hit shape boundary
        if len_prev + 1 != ndim {
            // new index can still be pushed
            idx_prev.push(0);
            return print_vec_with_layout_dfs_debug(c);
        } else {
            // last line

            // number of last dimension
            let nlast = *shape.last().unwrap();
            let stride_last = *stride.last().unwrap();

            // special case: zero shape quick return
            if nlast == 0 {
                return write!(fmt, "[]");
            }

            // prefix: "  [[["
            // count zeros from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "[")?;
            } else {
                let mut nbra = 1;
                for &idx in idx_prev.iter().rev() {
                    if idx == 0 {
                        nbra += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", " ".repeat(ndim - nbra))?;
                write!(fmt, "{:}", "[".repeat(nbra))?;
            }

            // values: " 1.23 4.56 ... 7.89 10.11"
            if nlast <= max_print.max(2 * min_print + 1) {
                // all elements in last dimension should be printed
                for i in 0..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Debug::fmt(&vec[offset_i], fmt)?;
                }
            } else {
                // only print the first/last min_print elements
                for i in 0..min_print {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Debug::fmt(&vec[offset_i], fmt)?;
                }
                write!(fmt, " ...")?;
                for i in (nlast - min_print)..nlast {
                    let offset_i = (offset as isize + i as isize * stride_last) as usize;
                    write!(fmt, " ")?;
                    Debug::fmt(&vec[offset_i], fmt)?;
                }
            };

            // suffix: "]]]"
            // count (if index = shape) from last element as numbers of "["
            if idx_prev.is_empty() {
                write!(fmt, "]")?;
            } else {
                let mut nket = 1;
                for i in (0..idx_prev.len()).rev() {
                    if idx_prev[i] == shape[i] - 1 {
                        nket += 1;
                    } else {
                        break;
                    }
                }
                write!(fmt, "{:}", "]".repeat(nket))?;
                if nket != ndim {
                    // last line should not add new-line character
                    if nket > 1 {
                        writeln!(fmt, "\n")?;
                    } else {
                        writeln!(fmt)?;
                    }
                }
            }

            // pop last index, increase previous index by 1 or skip
            if len_prev == 0 {
                // one-dimension exit
                return Ok(());
            } else {
                let p1_prev = idx_prev.pop().unwrap();
                let offset = offset as isize - p1_prev as isize * stride[len_prev - 1];
                if shape[len_prev - 1] < max_print
                    || p1_prev + min_print >= shape[len_prev - 1]
                    || p1_prev + 1 < min_print
                {
                    let p1 = p1_prev + 1;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs_debug(c);
                } else {
                    writeln!(fmt, "{}...", " ".repeat(len_prev))?;
                    let p1: usize = shape[len_prev - 1] - min_print;
                    idx_prev.push(p1);
                    let offset = offset + p1 as isize * stride[len_prev - 1];
                    let offset = offset as usize;
                    c.offset = offset;
                    return print_vec_with_layout_dfs_debug(c);
                }
            }
        }
    }
}

pub fn print_vec_with_layout_debug<T, D>(
    fmt: &mut core::fmt::Formatter<'_>,
    vec: &[T],
    layout: &Layout<D>,
    max_print: usize,
    min_print: usize,
) -> core::fmt::Result
where
    T: Clone + Debug,
    D: crate::layout::DimAPI,
{
    let idx_prev = vec![];
    let offset = layout.offset;
    let mut config =
        FnPrintVecWithLayout { vec, layout, offset, idx_prev, max_print, min_print, fmt };
    print_vec_with_layout_dfs_debug(&mut config)
}

impl<R, T, B, D> Debug for TensorAny<R, T, B, D>
where
    T: Clone + Debug,
    B: DeviceAPI<T> + Debug,
    D: DimAPI,
    R: DataAPI<Data = B::Raw>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "\n=== Debug Tensor Print ===")?;
        let vec = self.storage().to_cpu_vec().unwrap();
        let layout = &self.layout();
        let max_print = MAX_PRINT.load(Ordering::Relaxed);
        let min_print = MIN_PRINT.load(Ordering::Relaxed);
        print_vec_with_layout_debug(f, &vec, layout, max_print, min_print)?;
        writeln!(f)?;
        Debug::fmt(&self.device(), f)?;
        writeln!(f)?;
        Debug::fmt(&self.layout(), f)?;
        writeln!(f)?;
        let self_type = core::any::type_name::<Self>();
        writeln!(f, "Type: {}", self_type)?;
        writeln!(f, "==========================")
    }
}

/* #endregion */

#[cfg(test)]
mod playground {
    use super::*;

    #[derive(Debug)]
    struct VL<T, D>(Vec<T>, Layout<D>)
    where
        T: Clone + Display,
        D: DimAPI;

    impl<T, D> Display for VL<T, D>
    where
        T: Clone + Display,
        D: DimAPI,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let vec = self.0.as_slice();
            let layout = &self.1;
            let max_print = MAX_PRINT.load(Ordering::Relaxed);
            let min_print = MIN_PRINT.load(Ordering::Relaxed);
            print_vec_with_layout(f, vec, layout, max_print, min_print)
        }
    }

    #[test]
    fn playground() {
        use crate::layout::*;

        let mut s = String::new();

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [].into();
        println!("{:?}", VL(vec, layout));

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 0, 4].into();
        println!("{:}", VL(vec, layout));

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<_> = [2, 4, 0].into();
        println!("{:}", VL(vec, layout));
        println!("{:}", s);

        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout: Layout<Ix2> = [3, 5].into();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[4:13].reshape(3, 3)
        c = b.T[::2, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([2, 3], [2, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([4, 3], [1, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        /* Python code
        a = np.arange(15)
        b = a[2:14].reshape(3, 4)
        c = b.T[:, ::-1]
        */
        s.clear();
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let layout = Layout::<Ix2>::new([4, 3], [1, -4], 10).unwrap();
        println!("{:}", VL(vec, layout));

        s.clear();
        let vec = (0..1800).collect::<Vec<usize>>();
        let layout = Layout::<Ix3>::new([15, 12, 10], [1, 150, 15], 0).unwrap();
        println!("{:4}", VL(vec, layout));
    }
}
