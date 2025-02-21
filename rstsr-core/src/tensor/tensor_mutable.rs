//! Mutable tensor (either owned or mutable reference).

use crate::prelude_dev::*;

/// Mutable tensor (either owned or mutable reference).
///
/// This is mostly used for inplace operations as output.
///
/// It is designed similar to `TensorCow`.
/// However, if inplace operation is not convenient because of the layout
/// contiguous not fulfilled, a `ToBeCloned` variant is provided, where an owned
/// tensor with contiguous layout is generated. This is not the same to
/// `TensorCow`, where it only involves ownership conversion, but not layout
/// difference between two tensors. So this is defined as a special type.
pub enum TensorMutable<'a, T, B, D>
where
    B: DeviceRawAPI<T>,
    D: DimAPI,
{
    Owned(Tensor<T, B, D>),
    Mut(TensorMut<'a, T, B, D>),
    ToBeCloned(TensorMut<'a, T, B, D>, Tensor<T, B, D>),
}

impl<T, B, D> TensorViewAPI<T, B, D> for TensorMutable<'_, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn view(&self) -> TensorView<'_, T, B, D> {
        match self {
            TensorMutable::Owned(t) => t.view(),
            TensorMutable::Mut(t) => t.view(),
            TensorMutable::ToBeCloned(_, t) => t.view(),
        }
    }
}

impl<T, B, D> TensorViewMutAPI<T, B, D> for TensorMutable<'_, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    fn view_mut(&mut self) -> TensorViewMut<'_, T, B, D> {
        match self {
            TensorMutable::Owned(t) => t.view_mut(),
            TensorMutable::Mut(t) => t.view_mut(),
            TensorMutable::ToBeCloned(_, t) => t.view_mut(),
        }
    }
}

impl<T, B, D> TensorIntoOwnedAPI<T, B, D> for TensorMutable<'_, T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    fn into_owned(self) -> Tensor<T, B, D> {
        match self {
            TensorMutable::Owned(t) => t,
            TensorMutable::Mut(t) => t.into_owned(),
            TensorMutable::ToBeCloned(_, t) => t,
        }
    }
}

impl<T, B, D> TensorMutable<'_, T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    pub fn clone_to_mut(self) -> Self {
        match self {
            TensorMutable::ToBeCloned(mut arr_view, arr_owned) => {
                arr_view.assign(&arr_owned);
                TensorMutable::Mut(arr_view)
            },
            _ => self,
        }
    }

    pub fn into_reverse_axes(self) -> Self {
        match self {
            TensorMutable::Owned(t) => TensorMutable::Owned(t.into_reverse_axes()),
            TensorMutable::Mut(t) => TensorMutable::Mut(t.into_reverse_axes()),
            TensorMutable::ToBeCloned(t, t_owned) => {
                TensorMutable::ToBeCloned(t.into_reverse_axes(), t_owned.into_reverse_axes())
            },
        }
    }
}
