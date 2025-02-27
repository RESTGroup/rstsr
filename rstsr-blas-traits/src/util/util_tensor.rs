use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;

/* #region TensorOut */

pub type TensorMutable1<'a, T, B> = TensorMutable<'a, T, B, Ix1>;
pub type TensorMutable2<'a, T, B> = TensorMutable<'a, T, B, Ix2>;

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
