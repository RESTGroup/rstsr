use crate::prelude_dev::*;

// this value is used to determine whether to use contiguous inner iteration
const CONTIG_SWITCH: usize = 16;
// This value is used to determine when to use parallel iteration.
// Actual switch value is PARALLEL_SWITCH * RAYON_NUM_THREADS.
// Since current task is not intensive to each element, this value is large.
// 64 kB for f64
const PARALLEL_SWITCH: usize = 16384;
// For assignment, it is fully memory bounded; contiguous assignment is better
// handled by serial code. So we only do parallel in outer iteration
// (non-contiguous part).

#[duplicate_item(
    func_name
        func_name_serial
        TypeC TypeA Types
        func_clone
        bool_non_castable
    ;
    [assign_arbitary_cpu_rayon]
        [assign_arbitary_cpu_serial]
        [T] [T] [T: Clone + Send + Sync]
        [*ci = ai.clone()]
        [false]
    ;
    [assign_arbitary_uninit_cpu_rayon]
        [assign_arbitary_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone + Send + Sync]
        [ci.write(ai.clone())]
        [false]
    ;
    [assign_arbitary_promote_cpu_rayon]
        [assign_arbitary_promote_cpu_serial]
        [TC] [TA] [TC: Clone + Send + Sync, TA: Clone + Send + Sync + DTypePromotionAPI<TC>]
        [*ci = ai.clone().promote_astype()]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
    [assign_arbitary_uninit_promote_cpu_rayon]
        [assign_arbitary_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone + Send + Sync, TA: Clone + Send + Sync + DTypePromotionAPI<TC>]
        [ci.write(ai.clone().promote_astype())]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
)]
pub fn func_name<Types, DC, DA>(
    c: &mut [TypeC],
    lc: &Layout<DC>,
    a: &[TypeA],
    la: &Layout<DA>,
    default_order: FlagOrder,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    DC: DimAPI,
    DA: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return func_name_serial(c, lc, a, la, default_order);
    }

    if bool_non_castable {
        rstsr_raise!(
            RuntimeError,
            "Cannot promote from {} to {}",
            std::any::type_name::<TypeA>(),
            std::any::type_name::<TypeC>()
        )?;
    }

    // actual parallel iteration
    let contig = match default_order {
        RowMajor => lc.c_contig() && la.c_contig(),
        ColMajor => lc.f_contig() && la.f_contig(),
    };
    if contig {
        // contiguous case
        // we do not perform parallel for this case
        let offset_c = lc.offset();
        let offset_a = la.offset();
        let size = lc.size();
        c[offset_c..(offset_c + size)].iter_mut().zip(a[offset_a..(offset_a + size)].iter()).for_each(|(ci, ai)| {
            func_clone;
        });
        Ok(())
    } else {
        // determine order by layout preference
        let order = match default_order {
            RowMajor => TensorIterOrder::C,
            ColMajor => TensorIterOrder::F,
        };
        // generate col-major iterator
        let lc = translate_to_col_major_unary(lc, order)?;
        let la = translate_to_col_major_unary(la, order)?;
        // iterate and assign
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr() as *mut TypeC;
            let ci = &mut *c_ptr.add(idx_c);
            let ai = &a[idx_a];
            func_clone;
        };
        let task = || layout_col_major_dim_dispatch_par_2diff(&lc, &la, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

#[duplicate_item(
    func_name
        func_name_serial
        TypeC TypeA Types
        func_clone
        bool_non_castable
    ;
    [assign_cpu_rayon]
        [assign_cpu_serial]
        [T] [T] [T: Clone + Send + Sync]
        [*ci = ai.clone()]
        [false]
    ;
    [assign_uninit_cpu_rayon]
        [assign_uninit_cpu_serial]
        [MaybeUninit<T>] [T] [T: Clone + Send + Sync]
        [ci.write(ai.clone())]
        [false]
    ;
    [assign_promote_cpu_rayon]
        [assign_promote_cpu_serial]
        [TC] [TA] [TC: Clone + Send + Sync, TA: Clone + Send + Sync + DTypePromotionAPI<TC>]
        [*ci = ai.clone().promote_astype()]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
    [assign_uninit_promote_cpu_rayon]
        [assign_uninit_promote_cpu_serial]
        [MaybeUninit<TC>] [TA] [TC: Clone + Send + Sync, TA: Clone + Send + Sync + DTypePromotionAPI<TC>]
        [ci.write(ai.clone().promote_astype())]
        [!<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE]
    ;
)]
pub fn func_name<Types, D>(
    c: &mut [TypeC],
    lc: &Layout<D>,
    a: &[TypeA],
    la: &Layout<D>,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return func_name_serial(c, lc, a, la);
    }

    // re-align layouts
    let layouts_full = translate_to_col_major(&[lc, la], TensorIterOrder::K)?;
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous assign
        let lc = &layouts_full[0];
        let la = &layouts_full[1];
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr() as *mut TypeC;
            let ci = &mut *c_ptr.add(idx_c);
            let ai = &a[idx_a];
            func_clone;
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
        pool.map_or_else(task, |pool| pool.install(task))
    } else {
        // parallel for outer iteration
        let lc = &layouts_contig[0];
        let la = &layouts_contig[1];
        let func = |(idx_c, idx_a): (usize, usize)| unsafe {
            let c_ptr = c.as_ptr().add(idx_c) as *mut TypeC;
            (0..size_contig).for_each(|idx| {
                let ci = &mut *c_ptr.add(idx);
                let ai = &a[idx_a + idx];
                func_clone;
            })
        };
        let task = || layout_col_major_dim_dispatch_par_2(lc, la, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}

pub fn fill_cpu_rayon<T, D>(c: &mut [T], lc: &Layout<D>, fill: T, pool: Option<&ThreadPool>) -> Result<()>
where
    T: Clone + Send + Sync,
    D: DimAPI,
{
    fill_promote_cpu_rayon(c, lc, fill, pool)
}

pub fn fill_promote_cpu_rayon<TC, TA, D>(
    c: &mut [TC],
    lc: &Layout<D>,
    fill: TA,
    pool: Option<&ThreadPool>,
) -> Result<()>
where
    TC: Clone + Send + Sync,
    TA: Clone + Send + Sync + DTypePromotionAPI<TC>,
    D: DimAPI,
{
    // determine whether to use parallel iteration
    let size = lc.size();
    if size < PARALLEL_SWITCH || pool.is_none() {
        return fill_promote_cpu_serial(c, lc, fill);
    }

    if !<TA as DTypePromotionAPI<TC>>::CAN_ASTYPE {
        rstsr_raise!(
            RuntimeError,
            "Cannot promote from {} to {}",
            std::any::type_name::<TA>(),
            std::any::type_name::<TC>()
        )?;
    }

    // re-align layouts
    let layouts_full = [translate_to_col_major_unary(lc, TensorIterOrder::G)?];
    let layouts_full_ref = layouts_full.iter().collect_vec();
    let (layouts_contig, size_contig) = translate_to_col_major_with_contig(&layouts_full_ref);

    // actual parallel iteration
    if size_contig < CONTIG_SWITCH {
        // not possible for contiguous fill
        let lc = &layouts_full[0];
        let func = |idx_c| unsafe {
            let c_ptr = c.as_ptr() as *mut TC;
            *c_ptr.add(idx_c) = fill.clone().promote_astype();
        };
        let task = || layout_col_major_dim_dispatch_par_1(lc, func);
        pool.map_or_else(task, |pool| pool.install(task))
    } else {
        // parallel for outer iteration
        let lc = &layouts_contig[0];
        let func = |idx_c| unsafe {
            let c_ptr = c.as_ptr().add(idx_c) as *mut TC;
            (0..size_contig).for_each(|idx| {
                *c_ptr.add(idx) = fill.clone().promote_astype();
            })
        };
        let task = || layout_col_major_dim_dispatch_par_1(lc, func);
        pool.map_or_else(task, |pool| pool.install(task))
    }
}
