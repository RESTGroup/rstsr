use crate::traits_def::EigvalshAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_eigvalsh_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T::Real, DeviceFaer, Ix1>>
where
    T: ComplexField,
{
    // TODO: It seems faer is suspeciously slow on eigh function?
    // However, tests shows that results are correct.

    // set parallel mode
    let pool = a.device().get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(a.offset()),
            a.shape()[0],
            a.shape()[1],
            a.stride()[0],
            a.stride()[1],
        )
    };
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    // eigen value computation
    let result = faer_a
        .self_adjoint_eigenvalues(faer_uplo)
        .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;
    let eigenvalues = asarray((result, a.device())).into_dim::<Ix1>();

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    return Ok(eigenvalues);
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalshAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = faer_impl_eigvalsh_f(a.view(), uplo)?;
        let result = result.into_dim::<IxD>().into_dim::<D::SmallerOne>();
        Ok(result)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalshAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EigvalshAPI::<DeviceFaer>::eigvalsh_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EigvalshAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = Tensor<T::Real, DeviceFaer, D::SmallerOne>;
    fn eigvalsh_f(self) -> Result<Self::Out> {
        let a = self;
        EigvalshAPI::<DeviceFaer>::eigvalsh_f((a, None))
    }
}
