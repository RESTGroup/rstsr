use crate::prelude_dev::*;
use core::ops::Mul;
use num::Zero;
use rstsr_dtype_traits::ExtNum;

pub fn vecdot_naive_cpu_serial<TA, TB, TC, DA, DB, DC>(
    c: &mut [MaybeUninit<TC>],
    lc: &Layout<DC>,
    a: &[TA],
    la: &Layout<DA>,
    b: &[TB],
    lb: &Layout<DB>,
    axes_a: &[isize],
    axes_b: &[isize],
    order: FlagOrder,
) -> Result<()>
where
    TA: Clone,
    TB: Clone,
    TC: Clone + Zero,
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    TA: Mul<TB, Output = TC>,
    TA: ExtNum,
{
    let (las, lam) = la.dim_split_axes(axes_a)?;
    let (lbs, lbm) = lb.dim_split_axes(axes_b)?;
    // this check will also validate layout_col_major_dim_dispatch_2 that will raise
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;

    let lc = lc.to_dim::<IxD>()?;
    // broadcast layouts for computation
    let (lc_b, lam) = broadcast_layout_to_first(&lc, &lam, order)?;
    let (lc_b, lbm) = broadcast_layout_to_first(&lc_b, &lbm, order)?;
    rstsr_assert_eq!(lc_b, lc, InvalidLayout, "layout of c seems not broadcasted from a or b after axis sum")?;

    // accelarate flags generation
    // - flag_contig_s: summed part have contiguous memory access pattern
    let mut asc = vec![]; // [a]xes [s]ummed [c]ontig
    let (_, _, asc_a, _) = get_axes_composition(&las);
    let (_, _, asc_b, _) = get_axes_composition(&lbs);
    for (ic_a, ic_b) in izip!(asc_a, asc_b) {
        if ic_a == ic_b {
            asc.push(ic_a);
        } else {
            break;
        }
    }
    let flag_contig_s = !asc.is_empty();
    // - flag_contig_m: remaining part have contiguous memory access pattern
    let mut amc = vec![]; // [a]xes re[m]aining [c]ontig
    let (_, _, amc_a, _) = get_axes_composition(&lam);
    let (_, _, amc_b, _) = get_axes_composition(&lbm);
    let (_, _, amc_c, _) = get_axes_composition(&lc);
    for (ic_a, ic_b, ic_c) in izip!(amc_a, amc_b, amc_c) {
        if ic_a == ic_b && ic_b == ic_c {
            amc.push(ic_a);
        } else {
            break;
        }
    }
    let flag_contig_m = !amc.is_empty();

    // offset of layouts
    let offset_a = la.offset();
    let offset_b = lb.offset();

    if flag_contig_s {
        // number of elements as contiguous
        let n_contig = asc.iter().map(|&i| las.shape()[i]).product::<usize>();
        // translate layouts for discontiguous summed
        let asc = asc.iter().map(|&x| x as isize).collect_vec();
        let (_, lasd) = las.dim_split_axes(&asc)?;
        let (_, lbsd) = lbs.dim_split_axes(&asc)?;
        let layouts = translate_to_col_major(&[&lasd, &lbsd], TensorIterOrder::K)?;
        let (lasd, lbsd) = (&layouts[0], &layouts[1]);
        // translate layouts for remaining
        let layouts = translate_to_col_major(&[&lc, &lam, &lbm], TensorIterOrder::K)?;
        let (lc, lam, lbm) = (&layouts[0], &layouts[1], &layouts[2]);
        // perform computation
        layout_col_major_dim_dispatch_3(lc, lam, lbm, |(idx_c, idx_m_a, idx_m_b)| {
            let mut val_c: TC = Zero::zero();
            let stat = layout_col_major_dim_dispatch_2(lasd, lbsd, |(idx_s_a, idx_s_b)| {
                let idx_a = idx_m_a + idx_s_a - offset_a; // double count offset
                let idx_b = idx_m_b + idx_s_b - offset_b; // double count offset
                let val = unrolled_binary_reduce(
                    &a[idx_a..idx_a + n_contig],
                    &b[idx_b..idx_b + n_contig],
                    || Zero::zero(),
                    |acc, (x, y)| acc + x.ext_conj() * y,
                    |acc, v| acc + v,
                );
                val_c = val_c.clone() + val;
            });
            stat.rstsr_unwrap();
            c[idx_c].write(val_c);
        })
    } else if flag_contig_m {
        let n_contig = amc.iter().map(|&i| lam.shape()[i]).product::<usize>();
        // translate layouts for summed
        let layouts = translate_to_col_major(&[&las, &lbs], TensorIterOrder::K)?;
        let (las, lbs) = (&layouts[0], &layouts[1]);
        // translate layouts for discontiguous remaining
        let amc = amc.iter().map(|&x| x as isize).collect_vec();
        let (_, lamd) = lam.dim_split_axes(&amc)?;
        let (_, lbmd) = lbm.dim_split_axes(&amc)?;
        let (_, lcd) = lc.dim_split_axes(&amc)?;
        let layouts = translate_to_col_major(&[&lcd, &lamd, &lbmd], TensorIterOrder::K)?;
        let (lcd, lamd, lbmd) = (&layouts[0], &layouts[1], &layouts[2]);
        // perform computation
        const CHUNK: usize = 64;
        layout_col_major_dim_dispatch_3(lcd, lamd, lbmd, |(idx_c, idx_m_a, idx_m_b)| {
            let slc_c = &mut c[idx_c..idx_c + n_contig];
            // currently c has not been initialized, init here
            slc_c.iter_mut().for_each(|c| *c = MaybeUninit::new(Zero::zero()));
            for (ichunk, chunk_c) in slc_c.chunks_mut(CHUNK).enumerate() {
                let n_chunk = chunk_c.len();
                let idx_a_chunk = idx_m_a + ichunk * CHUNK;
                let idx_b_chunk = idx_m_b + ichunk * CHUNK;
                let stat = layout_col_major_dim_dispatch_2(las, lbs, |(idx_s_a, idx_s_b)| {
                    let idx_a = idx_a_chunk + idx_s_a - offset_a; // double count offset
                    let idx_b = idx_b_chunk + idx_s_b - offset_b; // double count offset
                    let chunk_a = &a[idx_a..idx_a + n_chunk];
                    let chunk_b = &b[idx_b..idx_b + n_chunk];
                    chunk_c.iter_mut().zip(chunk_a.iter().zip(chunk_b.iter())).for_each(|(c, (x, y))| unsafe {
                        let val = x.clone().ext_conj() * y.clone();
                        c.write(c.assume_init_read() + val);
                    });
                });
                stat.rstsr_unwrap();
            }
        })
    } else {
        // general case
        layout_col_major_dim_dispatch_3(&lc, &lam, &lbm, |(idx_c, idx_m_a, idx_m_b)| {
            if las.ndim() == 1 {
                let n_contract = las.shape()[0];
                let step_a = las.stride()[0];
                let step_b = lbs.stride()[0];
                let val_c = (0..n_contract).fold(Zero::zero(), |acc, i| {
                    let x = &a[(idx_m_a as isize + i as isize * step_a) as usize];
                    let y = &b[(idx_m_b as isize + i as isize * step_b) as usize];
                    acc + x.clone().ext_conj() * y.clone()
                });
                c[idx_c].write(val_c);
            } else {
                let mut val_c: TC = Zero::zero();
                let stat = layout_col_major_dim_dispatch_2(&las, &lbs, |(idx_s_a, idx_s_b)| {
                    let idx_a = idx_m_a + idx_s_a - offset_a; // double count offset
                    let idx_b = idx_m_b + idx_s_b - offset_b; // double count offset
                    val_c = val_c.clone() + a[idx_a].clone().ext_conj() * b[idx_b].clone();
                });
                stat.rstsr_unwrap();
                c[idx_c].write(val_c);
            }
        })
    }
}
