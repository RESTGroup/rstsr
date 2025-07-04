use crate::traits_def::DetAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use faer_ext::IntoFaer;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_det_f<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<T>
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

    // det computation
    let result = faer_a.determinant();

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
impl<ImplType> DetAPI<DeviceFaer> for Tr
where
    T: ComplexField,
    D: DimAPI,
{
    type Out = T;
    fn det_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(self.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = self;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_det_f(a_view)?;
        Ok(result)
    }
}
