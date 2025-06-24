use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

pub fn assign_arbitary_cpu_serial<T, DC, DA>(
    c: &mut [T],
    lc: &Layout<DC>,
    a: &[T],
    la: &Layout<DA>,
    order: FlagOrder,
) -> Result<()>
where
    T: Clone,
    DC: DimAPI,
    DA: DimAPI,
{
    let contig = match order {
        RowMajor => lc.c_contig() && la.c_contig(),
        ColMajor => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)]
            .iter_mut()
            .zip(a[offset_a..(offset_a + size)].iter())
            .for_each(|(ci, ai)| *ci = ai.clone());
    } else {
        // determine order by layout preference
        let order = match order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        layout_col_major_dim_dispatch_2diff(&lc, &la, |(idx_c, idx_a)| c[idx_c] = a[idx_a].clone())?;
    }
    Ok(())
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
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            c[idx_c..(idx_c + size_contig)]
                .iter_mut()
                .zip(a[idx_a..(idx_a + size_contig)].iter())
                .for_each(|(ci, ai)| *ci = ai.clone());
        })?;
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| c[idx_c] = a[idx_a].clone())?;
    }
    Ok(())
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
        layout_col_major_dim_dispatch_1(&layouts_contig[0], |idx_c| {
            for i in 0..size_contig {
                c[idx_c + i] = fill.clone();
            }
        })?;
    } else {
        layout_col_major_dim_dispatch_1(&layouts_full[0], |idx_c| {
            c[idx_c] = fill.clone();
        })?;
    }
    Ok(())
}
