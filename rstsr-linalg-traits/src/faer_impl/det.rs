use crate::traits_def::DetAPI;
use faer::prelude::*;
use faer::traits::ComplexField;
use rstsr_core::prelude_dev::*;
use rstsr_dtype_traits::ReImAPI;

pub fn faer_impl_det_f<T>(a: TensorView<'_, T, DeviceFaer, Ix2>) -> Result<T::Real>
where
    T: ComplexField + ReImAPI<Out = T::Real>,
{
    // set parallel mode
    let pool = a.device().get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    let faer_a = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(a.offset()),
            a.shape()[0],
            a.shape()[1],
            a.stride()[0],
            a.stride()[1],
        )
    };

    // det computation
    let result = faer_a.determinant();
    let result = result.real();

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    return Ok(result);
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> DetAPI<DeviceFaer> for Tr
where
    T: ComplexField + ReImAPI<Out = T::Real>,
    D: DimAPI,
{
    type Out = T::Real;
    fn det_f(self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            self.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = self;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_det_f(a_view.into())?;
        Ok(result)
    }
}
