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
    ImplType                                                       TrA                                TrB                              ;
   [T, D, Ra: DataAPI<Data = Vec<T>>, Rb: DataAPI<Data = Vec<T>>] [&TensorAny<Ra, T, DeviceFaer, D>] [&TensorAny<Rb, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [&TensorAny<R, T, DeviceFaer, D> ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>                             ] [TensorView<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                                                       ] [TensorView<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType                              TrA                                TrB                             ;
   ['b, T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ] [TensorMut<'b, T, DeviceFaer, D>];
   ['b, T, D,                          ] [TensorView<'_, T, DeviceFaer, D>] [TensorMut<'b, T, DeviceFaer, D>];
   [    T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ] [Tensor<T, DeviceFaer, D>       ];
   [    T, D,                          ] [TensorView<'_, T, DeviceFaer, D>] [Tensor<T, DeviceFaer, D>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (a, mut b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}

#[duplicate_item(
    ImplType                          TrA                               TrB                              ;
   [T, D, R: DataAPI<Data = Vec<T>>] [TensorMut<'_, T, DeviceFaer, D>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                          ] [TensorMut<'_, T, DeviceFaer, D>] [TensorView<'_, T, DeviceFaer, D>];
   [T, D, R: DataAPI<Data = Vec<T>>] [Tensor<T, DeviceFaer, D>       ] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D,                          ] [Tensor<T, DeviceFaer, D>       ] [TensorView<'_, T, DeviceFaer, D>];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (mut a, b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType                              TrA                               TrB                             ;
   ['b, T, D,                          ] [TensorMut<'_, T, DeviceFaer, D>] [TensorMut<'b, T, DeviceFaer, D>];
   [    T, D,                          ] [TensorMut<'_, T, DeviceFaer, D>] [Tensor<T, DeviceFaer, D>       ];
   ['b, T, D,                          ] [Tensor<T, DeviceFaer, D>       ] [TensorMut<'b, T, DeviceFaer, D>];
   [    T, D,                          ] [Tensor<T, DeviceFaer, D>       ] [Tensor<T, DeviceFaer, D>       ];
)]
impl<ImplType> SolveGeneralAPI<DeviceFaer> for (TrA, TrB)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = TrB;
    fn solve_general_f(self) -> Result<Self::Out> {
        let (mut a, mut b) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        let result = faer_impl_solve_general_f(a_view.into(), b_view.into())?;
        result.clone_to_mut();
        Ok(b)
    }
}
