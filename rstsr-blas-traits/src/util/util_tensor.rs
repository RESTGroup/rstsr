use num::complex::ComplexFloat;
use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;

/* #region TensorMutable */

pub type TensorMutable1<'a, T, B> = TensorMutable<'a, T, B, Ix1>;
pub type TensorMutable2<'a, T, B> = TensorMutable<'a, T, B, Ix2>;

pub fn overwritable_convert<T, B, D>(
    a: TensorReference<'_, T, B, D>,
) -> Result<TensorMutable<'_, T, B, D>>
where
    T: ComplexFloat,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, D>,
    D: DimAPI,
{
    let a = if a.is_ref() {
        TensorMutable::Owned(TensorView::from(a).into_contig_f(FlagOrder::default())?)
    } else {
        let a = TensorMut::from(a);
        if a.f_prefer() || a.c_prefer() {
            TensorMutable::Mut(a)
        } else {
            let a_buffer = a.to_contig_f(FlagOrder::default())?.into_owned();
            TensorMutable::ToBeCloned(a, a_buffer)
        }
    };
    Ok(a)
}

pub fn overwritable_convert_with_order<T, B, D>(
    a: TensorReference<'_, T, B, D>,
    order: FlagOrder,
) -> Result<TensorMutable<'_, T, B, D>>
where
    T: ComplexFloat,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, D>,
    D: DimAPI,
{
    let a = if a.is_ref() {
        TensorMutable::Owned(TensorView::from(a).into_contig_f(order)?)
    } else {
        let a = TensorMut::from(a);
        if (order == ColMajor && a.f_prefer()) || (order == RowMajor && a.c_prefer()) {
            TensorMutable::Mut(a)
        } else {
            let a_buffer = a.to_contig_f(order)?.into_owned();
            TensorMutable::ToBeCloned(a, a_buffer)
        }
    };
    Ok(a)
}

/* #endregion */

/* #region flip */

pub fn flip_trans<T, B>(
    order: TensorOrder,
    trans: FlagTrans,
    view: TensorView<'_, T, B, Ix2>,
    hermi: bool,
) -> Result<(FlagTrans, TensorCow<'_, T, B, Ix2>)>
where
    T: Clone,
    B: DeviceAPI<T>
        + DeviceConjAPI<T, Ix2, TOut = T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>,
{
    // row-major
    if (order == TensorOrder::C && view.c_prefer()) || (order == TensorOrder::F && view.f_prefer())
    {
        // tensor is already in the preferred order
        Ok((trans, view.into_cow()))
    } else {
        // otherwise, flip both the tensor and flag, and allocate new tensor if
        // necessary
        match trans {
            FlagTrans::N => Ok((trans.flip(hermi)?, match hermi {
                true => view.into_reverse_axes().change_prefer(order).conj().into_cow(),
                false => view.into_reverse_axes().change_prefer(order),
            })),
            FlagTrans::T => Ok((trans.flip(hermi)?, view.into_reverse_axes().change_prefer(order))),
            FlagTrans::C => Ok((
                trans.flip(hermi)?,
                view.into_reverse_axes().change_prefer(order).conj().into_cow(),
            )),
            _ => rstsr_invalid!(trans),
        }
    }
}

/* #endregion */
