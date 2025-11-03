use crate::prelude_dev::*;

impl<D> Debug for Layout<D>
where
    D: DimDevAPI,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let shape = self.shape().as_ref();
        let stride = self.stride().as_ref();
        let offset = self.offset();
        let is_c_contig = self.c_contig();
        let is_f_contig = self.f_contig();
        let is_c_prefer = self.c_prefer();
        let is_f_prefer = self.f_prefer();
        let mut contig = String::new();
        if self.size() == 0 {
            write!(contig, "Empty")?;
        } else {
            if is_c_contig {
                write!(contig, "C")?;
            }
            if is_c_prefer {
                write!(contig, "c")?;
            }
            if is_f_contig {
                write!(contig, "F")?;
            }
            if is_f_prefer {
                write!(contig, "f")?;
            }
            if contig.is_empty() {
                write!(contig, "Custom")?;
            }
        }
        write!(
            f,
            "{}-Dim{}, contiguous: {}\nshape: {:?}, stride: {:?}, offset: {}",
            self.ndim(),
            if self.shape().const_ndim().is_none() { " (dyn)" } else { "" },
            contig,
            shape,
            stride,
            offset,
        )?;
        Ok(())
    }
}
