use crate::traits_def::CholeskyAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_cholesky_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<Tensor<T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    if let Some(pool) = pool {
        faer::set_global_parallelism(Par::rayon(pool.current_num_threads()));
    }

    let uplo = uplo.unwrap_or(match a.device().default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let faer_a = a.into_faer();
    let faer_uplo = match uplo {
        Lower => faer::Side::Lower,
        Upper => faer::Side::Upper,
    };

    // llt computation
    let result = faer_a.llt(faer_uplo).map_err(|e| rstsr_error!(FaerError, "Faer cholesky error: {e}"))?;

    // faer always returns lower triangular matrix
    let result = match uplo {
        Lower => result.L(),
        Upper => result.L().transpose(),
    };
    // convert to rstsr tensor with certain layout
    let result = result.into_rstsr().into_contig(device.default_order());

    // restore parallel mode
    if pool.is_some() {
        faer::set_global_parallelism(faer_par_orig)
    }

    Ok(result)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for (Tr, Option<FlagUpLo>)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = faer_impl_cholesky_f(a.view(), uplo)?;
        let result = result.into_dim::<IxD>().into_dim::<D>();
        Ok(result)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for (Tr, FlagUpLo)
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        CholeskyAPI::<DeviceFaer>::cholesky_f((a, Some(uplo)))
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> CholeskyAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        CholeskyAPI::<DeviceFaer>::cholesky_f((a, None))
    }
}
