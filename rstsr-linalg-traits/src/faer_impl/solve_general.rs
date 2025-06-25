use crate::traits_def::SolveGeneralAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_solve_general_f<'b, T>(
    a: TensorReference<'_, T, DeviceFaer, Ix2>,
    b: TensorReference<'b, T, DeviceFaer, Ix2>,
) -> Result<TensorMutable<'b, T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    let faer_a = a.view().into_faer();

    // solve linear system
    let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SVD error: {e:?}"))?;

    // handle b for mutable
    let mut b = overwritable_convert(b)?;
    let b_view = b.view_mut().into_dim::<Ix2>();
    let faer_b = b_view.into_faer();

    svd_result.solve_in_place(faer_b);

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    Ok(b.clone_to_mut())
}

#[duplicate_item(
    ImplType                                                            TrA                                 TrB                              ;
   [T, DA, DB, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, DA>] [&TensorAny<Rb, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorView<'_, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, DA>] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                                                       ] [TensorView<'_, T, DeviceFaer, DA>] [TensorView<'_, T, DeviceFaer, DB>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, DB>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i((.., None)).into_dim::<Ix2>(),
            false => b.view().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        let result = result.into_owned().into_dim::<IxD>();
        match is_b_vec {
            true => Ok(result.into_shape(-1).into_dim::<DB>()),
            false => Ok(result.into_dim::<DB>()),
        }
    }
}

#[duplicate_item(
    ImplType                                   TrA                                 TrB                              ;
   ['b, T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [TensorMut<'b, T, DeviceFaer, DB>];
   ['b, T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, DA> ] [Tensor<T, DeviceFaer, DB>       ];
   [    T, DA, DB,                          ] [TensorView<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                               TrA                                TrB                               ;
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceFaer, DA>] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                          ] [TensorMut<'_, T, DeviceFaer, DA>] [TensorView<'_, T, DeviceFaer, DB>];
   [T, DA, DB, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceFaer, DA>       ] [&TensorAny<R, T, DeviceFaer, DB> ];
   [T, DA, DB,                          ] [Tensor<T, DeviceFaer, DA>       ] [TensorView<'_, T, DeviceFaer, DB>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, DB>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (mut a, b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i((.., None)).into_dim::<Ix2>(),
            false => b.view().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        let result = result.into_owned().into_dim::<IxD>();
        match is_b_vec {
            true => Ok(result.into_shape(-1).into_dim::<DB>()),
            false => Ok(result.into_dim::<DB>()),
        }
    }
}

#[duplicate_item(
    ImplType        TrA                               TrB                              ;
   ['b, T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [TensorMut<'_, T, DeviceFaer, DA>] [Tensor<T, DeviceFaer, DB>       ];
   ['b, T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [TensorMut<'b, T, DeviceFaer, DB>];
   [    T, DA, DB] [Tensor<T, DeviceFaer, DA>       ] [Tensor<T, DeviceFaer, DB>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    DA: DimAPI,
    DB: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (mut a, mut b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_pattern!(b.ndim(), 1..=2, InvalidLayout, "Currently we can only handle 1/2-D matrix.")?;
        let is_b_vec = b.ndim() == 1;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = match is_b_vec {
            true => b.i_mut((.., None)).into_dim::<Ix2>(),
            false => b.view_mut().into_dim::<Ix2>(),
        };
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}
