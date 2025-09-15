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

impl<T, B, D> TensorViewAPI for TensorMutable<'_, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

    fn view(&self) -> TensorView<'_, T, B, D> {
        match self {
            TensorMutable::Owned(t) => t.view(),
            TensorMutable::Mut(t) => t.view(),
            TensorMutable::ToBeCloned(_, t) => t.view(),
        }
    }
}

impl<T, B, D> TensorViewMutAPI for TensorMutable<'_, T, B, D>
where
    D: DimAPI,
    B: DeviceAPI<T>,
{
    type Type = T;
    type Backend = B;
    type Dim = D;

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
    B::Raw: Clone,
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

    pub fn f_prefer(&self) -> bool {
        self.view().f_prefer()
    }

    pub fn c_prefer(&self) -> bool {
        self.view().c_prefer()
    }

    pub fn f_contig(&self) -> bool {
        self.view().f_contig()
    }

    pub fn c_contig(&self) -> bool {
        self.view().c_contig()
    }
}

impl<T, B, D> TensorMutable<'_, T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
    B::Raw: Clone,
{
    pub fn to_owned(&self) -> Tensor<T, B, D> {
        match self {
            TensorMutable::Owned(t) => t.to_owned(),
            TensorMutable::Mut(t) => t.to_owned(),
            TensorMutable::ToBeCloned(_, t) => t.to_owned(),
        }
    }
}

impl<'a, T, B, D> TensorMutable<'a, T, B, D>
where
    T: Clone,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, D>,
{
    pub fn into_dim_f<D2>(self) -> Result<TensorMutable<'a, T, B, D2>>
    where
        D: DimIntoAPI<D2>,
        D2: DimAPI,
    {
        match self {
            TensorMutable::Owned(t) => Ok(TensorMutable::Owned(t.into_dim_f()?)),
            TensorMutable::Mut(t) => Ok(TensorMutable::Mut(t.into_dim_f()?)),
            TensorMutable::ToBeCloned(t, t_owned) => {
                Ok(TensorMutable::ToBeCloned(t.into_dim_f()?, t_owned.into_dim_f()?))
            },
        }
    }

    pub fn into_dim<D2>(self) -> TensorMutable<'a, T, B, D2>
    where
        D: DimIntoAPI<D2>,
        D2: DimAPI,
    {
        self.into_dim_f().unwrap()
    }
}
