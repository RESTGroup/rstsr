use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;

#[duplicate_item(
    func_name
        TypeC TypeA Types
        func_clone
        bool_non_castable
    ;
    [assign_arbitary_cpu_serial]
        [T] [T] [T: Clone]
        [*ci = ai.clone()]
        [false]
    ;
    [assign_arbitary_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone]
        [ci.write(ai.clone())]
        [false]
    ;
    [assign_arbitary_promote_cpu_serial]
        [TC] [TA] [TC: Clone, TA: Clone + DTypePromotionAPI<TC>]
        [*ci = ai.clone().promote_astype()]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
    [assign_arbitary_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone, TA: Clone + DTypePromotionAPI<TC>]
        [ci.write(ai.clone().promote_astype())]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
)]
pub fn func_name<Types, DC, DA>(
    c: &mut [TypeC],
    lc: &Layout<DC>,
    a: &[TypeA],
    la: &Layout<DA>,
    order: FlagOrder,
) -> Result<()>
where
    DC: DimAPI,
    DA: DimAPI,
{
    if bool_non_castable {
        rstsr_raise!(
            RuntimeError,
            "Cannot promote from {} to {}",
            std::any::type_name::<TypeA>(),
            std::any::type_name::<TypeC>()
        )?;
    }

    let contig = match order {
        RowMajor => lc.c_contig() && la.c_contig(),
        ColMajor => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].iter_mut().zip(a[offset_a..(offset_a + size)].iter()).for_each(|(ci, ai)| {
            func_clone;
        });
    } else {
        // determine order by layout preference
        let order = match order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        layout_col_major_dim_dispatch_2diff(&lc, &la, |(idx_c, idx_a)| {
            let ci = &mut c[idx_c];
            let ai = &a[idx_a];
            func_clone;
        })?;
    }
    Ok(())
}

#[duplicate_item(
    func_name
        TypeC TypeA Types
        func_clone
        bool_non_castable
    ;
    [assign_cpu_serial]
        [T] [T] [T: Clone]
        [*ci = ai.clone()]
        [false]
    ;
    [assign_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone]
        [ci.write(ai.clone())]
        [false]
    ;
    [assign_promote_cpu_serial]
        [TC] [TA] [TC: Clone, TA: Clone + DTypePromotionAPI<TC>]
        [*ci = ai.clone().promote_astype()]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
    [assign_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone, TA: Clone + DTypePromotionAPI<TC>]
        [ci.write(ai.clone().promote_astype())]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
)]
pub fn func_name<Types, D>(c: &mut [TypeC], lc: &Layout<D>, a: &[TypeA], la: &Layout<D>) -> Result<()>
where
    D: DimAPI,
{
    if bool_non_castable {
        rstsr_raise!(
            RuntimeError,
            "Cannot promote from {} to {}",
            std::any::type_name::<TypeA>(),
            std::any::type_name::<TypeC>()
        )?;
    }

    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    if size_contig >= CONTIG_SWITCH {
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            c[idx_c..(idx_c + size_contig)].iter_mut().zip(a[idx_a..(idx_a + size_contig)].iter()).for_each(
                |(ci, ai)| {
                    func_clone;
                },
            );
        })?;
    } else {
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        layout_col_major_dim_dispatch_2(lc, la, |(idx_c, idx_a)| {
            let ci = &mut c[idx_c];
            let ai = &a[idx_a];
            func_clone;
        })?;
    }
    Ok(())
}

pub fn fill_cpu_serial<T, D>(c: &mut [T], lc: &Layout<D>, fill: T) -> Result<()>
where
    T: Clone,
    D: DimAPI,
{
    fill_promote_cpu_serial(c, lc, fill)
}

pub fn fill_promote_cpu_serial<TC, TA, D>(c: &mut [TC], lc: &Layout<D>, fill: TA) -> Result<()>
where
    TA: Clone + DTypePromotionAPI<TC>,
    TC: Clone,
    D: DimAPI,
{
    if !<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE {
        rstsr_raise!(
            RuntimeError,
            "Cannot promote from {} to {}",
            std::any::type_name::<TA>(),
            std::any::type_name::<TC>()
        )?;
    }

    let fill = fill.clone().promote_astype();

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
