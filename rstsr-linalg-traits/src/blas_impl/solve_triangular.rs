use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                               ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceBLAS, DA>] [&TensorAny<Rb, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, DB>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i((.., None)).into_dim::<Ix2>(),
            false => b.view().into_dim::<Ix2>(),
        };
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        let result = result.into_owned().into_dim::<IxD>();
        match is_b_vec {
            true => Ok(result.into_shape(-1).into_dim::<DB>()),
            false => Ok(result.into_dim::<DB>()),
        }
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [TensorMut<'b, T, DeviceBLAS, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, DA> ] [Tensor<T, DeviceBLAS, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                               TrA                                TrB                               ;
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceBLAS, DA>] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                          ] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorView<'_, T, DeviceBLAS, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceBLAS, DA>       ] [&TensorAny<R, T, DeviceBLAS, DB> ];
   [T, DA, DB,                          ] [Tensor<T, DeviceBLAS, DA>       ] [TensorView<'_, T, DeviceBLAS, DB>];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, DB>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (mut a, b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i((.., None)).into_dim::<Ix2>(),
            false => b.view().into_dim::<Ix2>(),
        };
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        let result = result.into_owned().into_dim::<IxD>();
        match is_b_vec {
            true => Ok(result.into_shape(-1).into_dim::<DB>()),
            false => Ok(result.into_dim::<DB>()),
        }
    }
}

#[duplicate_item(
    ImplType        TrA                                TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceBLAS, DA>] [Tensor<T, DeviceBLAS, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [TensorMut<'b, T, DeviceBLAS, DB>];
   [    T, DA, DB] [Tensor<T, DeviceBLAS, DA>       ] [Tensor<T, DeviceBLAS, DB>       ];
)]
impl<ImplType> SolveTriangularAPI<DeviceBLAS> for (TrA, TrB, Option<FlagUpLo>)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TrB;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (mut a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
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
