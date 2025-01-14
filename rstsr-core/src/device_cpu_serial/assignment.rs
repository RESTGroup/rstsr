use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

pub fn assign_arbitary_cpu_serial<T, DC, DA>(
    c: &mut [T],
    lc: &Layout<DC>,
    a: &[T],
    la: &Layout<DA>,
) -> Result<()>
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    let contig = match TensorOrder::default() {
        TensorOrder::C => lc.c_contig() && la.c_contig(),
        TensorOrder::F => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].clone_from_slice(&a[offset_a..(offset_a + size)]);
    } else {
        // determine order by layout preference
        let order = match TensorOrder::default() {
            TensorOrder::C => TensorIterOrder::C,
            TensorOrder::F => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        let iter_c = IterLayoutColMajor::new(&lc)?;
        let iter_a = IterLayoutColMajor::new(&la)?;
        // iterate and assign
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            c[idx_c] = a[idx_a].clone();
        }
    }
    return Ok(());
}

pub fn assign_cpu_serial<T, D>(c: &mut [T], lc: &Layout<D>, a: &[T], la: &Layout<D>) -> Result<()>
where
    T: Clone,
    D: DimAPI,
{
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig >= CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_contig[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            c[idx_c..(idx_c + size_contig)].clone_from_slice(&a[idx_a..(idx_a + size_contig)]);
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        let iter_a = IterLayoutColMajor::new(&layouts_full[1])?;
        for (idx_c, idx_a) in izip!(iter_c, iter_a) {
            c[idx_c] = a[idx_a].clone();
        }
    }
    return Ok(());
}

pub fn fill_cpu_serial<T, D>(c: &mut [T], lc: &Layout<D>, fill: T) -> Result<()>
where
    T: Clone,
    D: DimAPI,
{
    let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig > CONTIG_SWITCH {
        let iter_c = IterLayoutColMajor::new(&layouts_contig[0])?;
        for idx_c in iter_c {
            for i in 0..size_contig {
                c[idx_c + i] = fill.clone();
            }
        }
    } else {
        let iter_c = IterLayoutColMajor::new(&layouts_full[0])?;
        for idx_c in iter_c {
            c[idx_c] = fill.clone();
        }
    }
    return Ok(());
}

impl<T, DC, DA> OpAssignArbitaryAPI<T, DC, DA> for DeviceCpuSerial
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    fn assign_arbitary(
        &self,
        c: &mut Vec<T>,
        lc: &Layout<DC>,
        a: &Vec<T>,
        la: &Layout<DA>,
    ) -> Result<()> {
        return assign_arbitary_cpu_serial(c, lc, a, la);
    }
}

impl<T, D> OpAssignAPI<T, D> for DeviceCpuSerial
where
    T: Clone,
    D: DimAPI,
{
    fn assign(&self, c: &mut Vec<T>, lc: &Layout<D>, a: &Vec<T>, la: &Layout<D>) -> Result<()> {
        return assign_cpu_serial(c, lc, a, la);
    }

    fn fill(&self, c: &mut Vec<T>, lc: &Layout<D>, fill: T) -> Result<()> {
        return fill_cpu_serial(c, lc, fill);
    }
}
