use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

#[duplicate_item(
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, D>] [&TensorAny<Rb, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorCow<'_, T, DeviceBLAS, D> ] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorView<'_, T, DeviceBLAS, D>] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                                                       ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        return Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>());
    }
}

#[duplicate_item(
    ImplType                              TrA                                TrB                             ;
   ['b, T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ] [TensorMut<'b, T, DeviceBLAS, D>];
   ['b, T, D,                          ] [TensorView<'_, T, DeviceBLAS, D>] [TensorMut<'b, T, DeviceBLAS, D>];
   ['b, T, D,                          ] [TensorCow<'_, T, DeviceBLAS, D> ] [TensorMut<'b, T, DeviceBLAS, D>];
   [    T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ] [Tensor<T, DeviceBLAS, D>       ];
   [    T, D,                          ] [TensorView<'_, T, DeviceBLAS, D>] [Tensor<T, DeviceBLAS, D>       ];
   [    T, D,                          ] [TensorCow<'_, T, DeviceBLAS, D> ] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                          TrA                               TrB                              ;
   [T, D, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceBLAS, D>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                          ] [TensorMut<'_, T, DeviceBLAS, D>] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                          ] [TensorMut<'_, T, DeviceBLAS, D>] [TensorCow<'_, T, DeviceBLAS, D> ];
   [T, D, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceBLAS, D>       ] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D,                          ] [Tensor<T, DeviceBLAS, D>       ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D,                          ] [Tensor<T, DeviceBLAS, D>       ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (mut a, b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType                              TrA                               TrB                             ;
   ['b, T, D,                          ] [TensorMut<'_, T, DeviceBLAS, D>] [TensorMut<'b, T, DeviceBLAS, D>];
   [    T, D,                          ] [TensorMut<'_, T, DeviceBLAS, D>] [Tensor<T, DeviceBLAS, D>       ];
   ['b, T, D,                          ] [Tensor<T, DeviceBLAS, D>       ] [TensorMut<'b, T, DeviceBLAS, D>];
   [    T, D,                          ] [Tensor<T, DeviceBLAS, D>       ] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (mut a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct             args_tuple     internal_tuple     ;
   [(TrA, TrB, FlagUpLo)] [(a, b, uplo)] [(a, b, Some(uplo))];
   [(TrA, TrB,         )] [(a, b,     )] [(a, b, None      )];
)]
impl<TrA, TrB> SolveTriangularAPI<DeviceBLAS> for ImplStruct
where
    (TrA, TrB, Option<FlagUpLo>): SolveTriangularAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, Option<FlagUpLo>) as SolveTriangularAPI<DeviceBLAS>>::Out;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        SolveTriangularAPI::<DeviceBLAS>::solve_triangular_f(internal_tuple)
    }
}

/* #endregion */
