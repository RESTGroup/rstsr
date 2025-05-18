use crate::traits_def::{PinvAPI, PinvResult};
use faer::prelude::*;
use faer::traits::ComplexField;
use num::{Float, FromPrimitive, Num, Zero};
use rstsr_core::prelude_dev::*;

pub fn faer_impl_pinv_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<PinvResult<Tensor<T, DeviceFaer, Ix2>>>
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Send + Sync,
{
    // set parallel mode
    let pool = a.device().get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    // compute rcond value
    let atol = atol.unwrap_or(T::Real::zero());
    let rtol = rtol.unwrap_or({
        let [m, n]: [usize; 2] = *a.shape();
        let mnmax = T::Real::from_usize(Ord::max(m, n)).unwrap();
        mnmax * T::Real::epsilon()
    });

    // transform to faer matrix
    let faer_a = unsafe {
        MatRef::from_raw_parts(
            a.as_ptr().add(a.offset()),
            a.shape()[0],
            a.shape()[1],
            a.stride()[0],
            a.stride()[1],
        )
    };

    // svd computation
    let svd_result =
        faer_a.thin_svd().map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors
    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    // compute pinv
    let s = s.mapv(|x| T::real_part_impl(&x));
    let maxs = s.raw().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let val = atol + rtol * maxs;
    let rank = s.raw().iter().take_while(|&&x| x > val).count();
    let mut u = u.into_slice((.., ..rank));
    u /= s.i((None, ..rank));
    let a_pinv = v.i((.., ..rank)) % u.t();
    let pinv = a_pinv.mapv(|x| T::conj_impl(&x)).into_dim::<Ix2>();

    // restore parallel mode
    faer::set_global_parallelism(faer_par_orig);

    Ok(PinvResult { pinv, rank })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> PinvAPI<DeviceFaer> for (Tr, T::Real, T::Real)
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Zero + Send + Sync,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_pinv_f(a_view, Some(atol), Some(rtol))?;
        // convert dimensions
        Ok(PinvResult { pinv: result.pinv.into_dim::<IxD>().into_dim::<D>(), rank: result.rank })
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType> PinvAPI<DeviceFaer> for Tr
where
    T: ComplexField + DivAssign<T::Real> + Num + Send + Sync + 'static,
    T::Real: Float + FromPrimitive + Zero + Send + Sync,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, D>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_pinv_f(a_view, None, None)?;
        // convert dimensions
        Ok(PinvResult { pinv: result.pinv.into_dim::<IxD>().into_dim::<D>(), rank: result.rank })
    }
}
