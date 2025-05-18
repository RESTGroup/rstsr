use crate::traits_def::{PinvAPI, PinvResult};
use faer::linalg::solvers::Svd;
use faer::prelude::*;
use faer::traits::ComplexField;
use num::{complex::ComplexFloat, Bounded, Float, FromPrimitive, Zero};
use rstsr_core::prelude_dev::*;
use rstsr_dtype_traits::MinMaxAPI;

pub fn faer_impl_pinv_f<T>(
    a: TensorView<'_, T, DeviceFaer, Ix2>,
    atol: Option<<T as ComplexField>::Real>,
    rtol: Option<<T as ComplexField>::Real>,
) -> Result<PinvResult<Tensor<T, DeviceFaer, Ix2>>>
where
    T: ComplexField
        + ComplexFloat<Real = <T as ComplexField>::Real>
        + DivAssign<<T as ComplexField>::Real>
        + Send
        + Sync
        + 'static,
    <T as ComplexField>::Real: Float + FromPrimitive + Zero + Send + Sync + MinMaxAPI + Bounded,
{
    // set parallel mode
    let pool = a.device().get_current_pool();
    let faer_par_orig = faer::get_global_parallelism();
    let faer_par = pool.map_or(Par::Seq, |pool| Par::rayon(pool.current_num_threads()));
    faer::set_global_parallelism(faer_par);

    // compute rcond value
    let atol = atol.unwrap_or(<T as ComplexField>::Real::zero());
    let rtol = rtol.unwrap_or({
        let [m, n]: [usize; 2] = *a.shape();
        let mnmax = <T as ComplexField>::Real::from_usize(Ord::max(m, n)).unwrap();
        mnmax * <T as ComplexField>::Real::epsilon()
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
        Svd::new_thin(faer_a).map_err(|e| rstsr_error!(FaerError, "Faer SvD error: {e:?}"))?;
    let (u, s, v) = (svd_result.U(), svd_result.S(), svd_result.V());

    // return to rstsr tensors
    let u = u.into_rstsr().into_owned();
    let s = s.column_vector().into_rstsr();
    let v = v.into_rstsr();

    // compute pinv
    let s = s.mapv(|x| x.re());
    let maxs = s.max_all();
    let val = atol + rtol * maxs;
    let rank = s.raw().iter().take_while(|&&x| x > val).count();
    let mut u = u.into_slice((.., ..rank));
    u /= s.i((None, ..rank));
    let a_pinv = v.i((.., ..rank)) % u.t();
    let pinv = a_pinv.conj().into_dim::<Ix2>();

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
impl<ImplType> PinvAPI<DeviceFaer> for (Tr, <T as ComplexField>::Real, <T as ComplexField>::Real)
where
    T: ComplexField
        + ComplexFloat<Real = <T as ComplexField>::Real>
        + DivAssign<<T as ComplexField>::Real>
        + Send
        + Sync
        + 'static,
    <T as ComplexField>::Real: Float + FromPrimitive + Zero + Send + Sync + MinMaxAPI + Bounded,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, Ix2>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let (a, atol, rtol) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_pinv_f(a_view, Some(atol), Some(rtol))?;
        Ok(result)
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
    T: ComplexField
        + ComplexFloat<Real = <T as ComplexField>::Real>
        + DivAssign<<T as ComplexField>::Real>
        + Send
        + Sync
        + 'static,
    <T as ComplexField>::Real: Float + FromPrimitive + Zero + Send + Sync + MinMaxAPI + Bounded,
    D: DimAPI,
{
    type Out = PinvResult<Tensor<T, DeviceFaer, Ix2>>;
    fn pinv_f(self) -> Result<Self::Out> {
        let a = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let result = faer_impl_pinv_f(a_view, None, None)?;
        Ok(result)
    }
}
