use crate::traits_def::InvAPI;
use faer::linalg::solvers::DenseSolveCore;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_inv_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
) -> Result<Tensor<T, DeviceFaer, Ix2>>
where
    T: ComplexField,
{
    // set parallel mode
    let device = a.device().clone();
    let pool = device.get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    let faer_a = a.into_faer();

    // det computation
    let svd_result = faer_a.svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let result = svd_result.inverse();

    // convert to rstsr tensor with certain layout
    let result = result.as_ref().into_rstsr().into_contig(device.default_order());

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    Ok(result)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> InvAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = Tensor<T, DeviceFaer, D>;
    fn inv_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = self;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_inv_f(a_view)?;
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}
