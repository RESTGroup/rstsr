use crate::prelude_dev::*;

impl<S> TensorBase<S, Ix2> {
    pub fn nrow(&self) -> usize {
        self.shape()[0]
    }

    pub fn ncol(&self) -> usize {
        self.shape()[1]
    }

    /// Leading dimension in row-major order case.
    ///
    /// This function will not return any value if the layout is not row-major.
    pub fn ld_row(&self) -> Option<usize> {
        if !self.c_prefer() {
            // leading dimension is only defined if not c-prefer
            return None;
        } else if self.shape()[0] == 1 {
            // col-vector, leading dimension must be larger than dimension of col
            return Some(self.shape()[1]);
        } else {
            // usual definition that leading dimension is stride of row
            return Some(self.stride()[0] as usize);
        }
    }

    /// Leading dimension in column-major order case.
    ///
    /// This function will not return any value if the layout is not
    /// column-major.
    pub fn ld_col(&self) -> Option<usize> {
        if self.c_prefer() {
            // leading dimension is only defined if not f-prefer
            return None;
        } else if self.shape()[1] == 1 {
            // row-vector, leading dimension must be larger than dimension of row
            return Some(self.shape()[0]);
        } else {
            // usual definition that leading dimension is stride of col
            return Some(self.stride()[1] as usize);
        }
    }

    /// Leading dimension by order.
    pub fn ld(&self, order: FlagOrder) -> Option<usize> {
        match order {
            ColMajor => self.ld_col(),
            RowMajor => self.ld_row(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn playground() {
        let l = Layout::new([6, 1], [100, 10], 0).unwrap();
        println!("{:?}", l.f_prefer());
        println!("{:?}", l.c_prefer());
        println!("{l:?}");
    }
}
