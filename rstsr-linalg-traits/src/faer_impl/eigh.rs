use crate::traits_def::{EighAPI, EighResult};
use faer::prelude::*;
use faer::traits::ComplexField;
use rstsr_core::prelude_dev::*;
use rstsr_dtype_traits::ReImAPI;

pub fn faer_impl_eigh_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<(Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>)>
where
    T: ComplexField + ReImAPI<Out = T::Real>,
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
        .self_adjoint_eigen(faer_uplo)
        .map_err(|e| rstsr_error!(FaerError, "Faer SelfAdjointEigen error: {e:?}"))?;

    // convert eigenvalues to real
    let eigenvalues: TensorView<T, DeviceFaer, _> = result.S().column_vector().into_rstsr();
    let eigenvalues = eigenvalues.real();
    let eigenvectors = result.U().into_rstsr().into_contig(a.device().default_order());

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    return Ok((eigenvalues, eigenvectors));
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = faer_impl_eigh_f(a.view(), uplo)?;
        let result = EighResult {
            eigenvalues: result.0.into_dim::<IxD>().into_dim::<D::SmallerOne>(),
            eigenvectors: result.1.into_owned().into_dim::<IxD>().into_dim::<D>(),
        };
        Ok(result)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        EighAPI::<DeviceFaer>::eigh_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> EighAPI<DeviceFaer> for Tr
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out = EighResult<Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn eigh_f(self) -> Result<Self::Out> {
        let a = self;
        EighAPI::<DeviceFaer>::eigh_f((a, None))
    }
}
