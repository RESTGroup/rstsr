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
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
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
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, mut b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
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
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (mut a, b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
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
impl<ImplType> SolveSymmetricAPI<DeviceBLAS> for (TrA, TrB, bool, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (mut a, mut b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
     ImplStruct                   args_tuple            internal_tuple            ;
    [(TrA, TrB, bool, FlagUpLo)] [(a, b, hermi, uplo)] [(a, b, hermi, Some(uplo))];
    [(TrA, TrB, bool,         )] [(a, b, hermi,     )] [(a, b, hermi, None      )];
    [(TrA, TrB,       FlagUpLo)] [(a, b,        uplo)] [(a, b, true , Some(uplo))];
    [(TrA, TrB                )] [(a, b,            )] [(a, b, true , None      )];
)]
impl<TrA, TrB> SolveSymmetricAPI<DeviceBLAS> for ImplStruct
where
    (TrA, TrB, bool, Option<FlagUpLo>): SolveSymmetricAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, bool, Option<FlagUpLo>) as SolveSymmetricAPI<DeviceBLAS>>::Out;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        SolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f(internal_tuple)
    }
}

/* #endregion */
