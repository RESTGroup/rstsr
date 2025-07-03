use crate::traits_def::{SVDResult, SVDAPI};
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_svd_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    full_matrices: bool,
) -> Result<SVDResult<Tensor<T, DeviceFaer, Ix2>, Tensor<T::Real, DeviceFaer, Ix1>, Tensor<T, DeviceFaer, Ix2>>>
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

    let faer_a = a.into_faer();

    // svd computation
    let svd_result = match full_matrices {
        true => faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
        false => faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?,
    };
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors
    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    let result = SVDResult {
        u: u.into_contig(device.default_order()),
        s: s.mapv(|v| T::real_part_impl(&v)).into_contig(device.default_order()),
        vt: v.into_reverse_axes().into_contig(device.default_order()),
    };

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
impl<ImplType> SVDAPI<DeviceFaer> for (Tr, bool)
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out =
        SVDResult<Tensor<T, DeviceFaer, D>, Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        let (a, full_matrices) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_svd_f(a_view, full_matrices)?;
        // convert dimensions
        Ok(SVDResult {
            u: result.u.into_dim::<IxD>().into_dim::<D>(),
            s: result.s.into_dim::<IxD>().into_dim::<D::SmallerOne>(),
            vt: result.vt.into_dim::<IxD>().into_dim::<D>(),
        })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> SVDAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
{
    type Out =
        SVDResult<Tensor<T, DeviceFaer, D>, Tensor<T::Real, DeviceFaer, D::SmallerOne>, Tensor<T, DeviceFaer, D>>;
    fn svd_f(self) -> Result<Self::Out> {
        SVDAPI::<DeviceFaer>::svd_f((self, true))
    }
}
