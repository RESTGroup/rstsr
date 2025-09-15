//! Trait and implementations for list of tensor views.

use crate::prelude_dev::*;

/// Trait for list of tensor views.
pub trait TensorViewListAPI<T, B>
where
    B: DeviceRawAPI<T>,
{
    /// Function to get a list of tensor views.
    ///
    /// This is used to generalize functions that take multiple tensors as input.
    /// The input can be
    ///
    /// - a slice of tensors (either owned or reference);
    /// - a tuple of tensors (up to 9 elements).
    ///
    /// Note that
    ///
    /// - all tensors must have the same data type and backend;
    /// - the dimension can be different, but will be converted to `IxD` (dynamic dimension).
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>>;
}

impl<R, T, B> TensorViewListAPI<T, B> for &[TensorAny<R, T, B, IxD>]
where
    B: DeviceAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        self.iter().map(|t| t.view()).collect()
    }
}

impl<R, T, B> TensorViewListAPI<T, B> for &[&TensorAny<R, T, B, IxD>]
where
    B: DeviceAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        self.iter().map(|t| t.view()).collect()
    }
}

impl<T, B, A1> TensorViewListAPI<T, B> for (A1,)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![self.0.view().into_dim::<IxD>()]
    }
}

impl<T, B, A1, A2> TensorViewListAPI<T, B> for (A1, A2)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![self.0.view().into_dim::<IxD>(), self.1.view().into_dim::<IxD>()]
    }
}

impl<T, B, A1, A2, A3> TensorViewListAPI<T, B> for (A1, A2, A3)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![self.0.view().into_dim::<IxD>(), self.1.view().into_dim::<IxD>(), self.2.view().into_dim::<IxD>()]
    }
}

impl<T, B, A1, A2, A3, A4> TensorViewListAPI<T, B> for (A1, A2, A3, A4)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
        ]
    }
}

impl<T, B, A1, A2, A3, A4, A5> TensorViewListAPI<T, B> for (A1, A2, A3, A4, A5)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A5: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
            self.4.view().into_dim::<IxD>(),
        ]
    }
}

impl<T, B, A1, A2, A3, A4, A5, A6> TensorViewListAPI<T, B> for (A1, A2, A3, A4, A5, A6)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A5: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A6: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
            self.4.view().into_dim::<IxD>(),
            self.5.view().into_dim::<IxD>(),
        ]
    }
}

impl<T, B, A1, A2, A3, A4, A5, A6, A7> TensorViewListAPI<T, B> for (A1, A2, A3, A4, A5, A6, A7)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A5: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A6: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A7: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
            self.4.view().into_dim::<IxD>(),
            self.5.view().into_dim::<IxD>(),
            self.6.view().into_dim::<IxD>(),
        ]
    }
}

impl<T, B, A1, A2, A3, A4, A5, A6, A7, A8> TensorViewListAPI<T, B> for (A1, A2, A3, A4, A5, A6, A7, A8)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A5: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A6: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A7: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A8: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
            self.4.view().into_dim::<IxD>(),
            self.5.view().into_dim::<IxD>(),
            self.6.view().into_dim::<IxD>(),
            self.7.view().into_dim::<IxD>(),
        ]
    }
}

impl<T, B, A1, A2, A3, A4, A5, A6, A7, A8, A9> TensorViewListAPI<T, B> for (A1, A2, A3, A4, A5, A6, A7, A8, A9)
where
    A1: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A2: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A3: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A4: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A5: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A6: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A7: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A8: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    A9: TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    B: DeviceAPI<T>,
{
    fn view_list(&'_ self) -> Vec<TensorView<'_, T, B, IxD>> {
        vec![
            self.0.view().into_dim::<IxD>(),
            self.1.view().into_dim::<IxD>(),
            self.2.view().into_dim::<IxD>(),
            self.3.view().into_dim::<IxD>(),
            self.4.view().into_dim::<IxD>(),
            self.5.view().into_dim::<IxD>(),
            self.6.view().into_dim::<IxD>(),
            self.7.view().into_dim::<IxD>(),
            self.8.view().into_dim::<IxD>(),
        ]
    }
}
