use rstsr_core::prelude::*;
use rstsr_core::prelude_dev::*;

/* #region TensorMutable */

pub type TensorMutable1<'a, T, B> = TensorMutable<'a, T, B, Ix1>;
pub type TensorMutable2<'a, T, B> = TensorMutable<'a, T, B, Ix2>;

/// Convert a tensor reference to a mutable tensor with optimal layout for LAPACK.
///
/// This function converts the input tensor to a contiguous layout suitable for
/// LAPACK operations:
/// - F-prefer (column-major prefer) tensors are converted to ColMajor (stride=[1, ld])
/// - C-prefer (row-major prefer) tensors are converted to RowMajor (stride=[ld, 1])
///
/// The conversion is "overwritable" meaning:
/// - If the input is a reference (read-only), a new owned tensor is created
/// - If the input is already mutable and contiguous, it's returned as-is
/// - If the input is mutable but not contiguous, it's converted with data cloning
///
/// # Returns
///
/// A `TensorMutable` that can be used for in-place LAPACK operations.
/// The caller should check `f_prefer()` or `c_prefer()` to determine the actual
/// layout after conversion.
///
/// # Example
///
/// ```ignore
/// let mut a = overwritable_convert(a)?;
/// let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };
/// let lda = a.view().ld(order).unwrap();
/// // Now call LAPACK with the correct order and lda
/// ```
pub fn overwritable_convert<T, B, D>(a: TensorReference<'_, T, B, D>) -> Result<TensorMutable<'_, T, B, D>>
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
    D: DimAPI,
{
    let order = match (a.f_prefer(), a.c_prefer()) {
        (true, false) => ColMajor,
        (false, true) => RowMajor,
        _ => a.device().default_order(),
    };
    let a = if a.is_ref() {
        TensorMutable::Owned(TensorView::from(a).into_contig_f(order)?)
    } else {
        let a = TensorMut::from(a);
        if a.f_prefer() || a.c_prefer() {
            TensorMutable::Mut(a)
        } else {
            let a_buffer = a.to_contig_f(order)?.into_owned();
            TensorMutable::ToBeCloned(a, a_buffer)
        }
    };
    Ok(a)
}

/// Convert a tensor reference to a mutable tensor with a specified layout order.
///
/// Unlike `overwritable_convert`, this function forces conversion to a specific
/// order (RowMajor or ColMajor), potentially requiring a transpose operation.
///
/// **Note**: For most LAPACK drivers, prefer `overwritable_convert` instead,
/// as it automatically selects the optimal layout without unnecessary conversions.
/// Use this function only when you specifically need a particular order.
///
/// # Arguments
///
/// * `a` - Input tensor reference
/// * `order` - Target order (RowMajor or ColMajor)
///
/// # Returns
///
/// A `TensorMutable` with the specified order.
pub fn overwritable_convert_with_order<T, B, D>(
    a: TensorReference<'_, T, B, D>,
    order: FlagOrder,
) -> Result<TensorMutable<'_, T, B, D>>
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D> + OpAssignAPI<T, D>,
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
    order: FlagOrder,
    trans: FlagTrans,
    view: TensorView<'_, T, B, Ix2>,
    hermi: bool,
) -> Result<(FlagTrans, TensorCow<'_, T, B, Ix2>)>
where
    T: Clone,
    B: DeviceAPI<T>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, Ix2, Ix2>
        + OpAssignAPI<T, Ix2>
        + OpConjAPI<T, Ix2, TOut = T>,
{
    // row-major
    if (order == FlagOrder::C && view.c_prefer()) || (order == FlagOrder::F && view.f_prefer()) {
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
            FlagTrans::C => Ok((trans.flip(hermi)?, view.into_reverse_axes().change_prefer(order).conj().into_cow())),
            _ => rstsr_invalid!(trans),
        }
    }
}

/* #endregion */
