use rstsr_blas_traits::lapack_solve::getrf::*;
use rstsr_blas_traits::lapack_solve::getri::*;
use rstsr_blas_traits::prelude_dev::*;
use rstsr_core::error::Result;
use rstsr_core::prelude_dev::*;

pub trait LinalgInvAPI<Inp>: Sized {
    type Out;
    fn inv_f(args: Self) -> Result<Self::Out>;
    fn inv(args: Self) -> Self::Out {
        Self::inv_f(args).unwrap()
    }
}

pub fn inv_f<Args, Inp>(args: Args) -> Result<<Args as LinalgInvAPI<Inp>>::Out>
where
    Args: LinalgInvAPI<Inp>,
{
    Args::inv_f(args)
}

pub fn inv<Args, Inp>(args: Args) -> <Args as LinalgInvAPI<Inp>>::Out
where
    Args: LinalgInvAPI<Inp>,
{
    Args::inv(args)
}

/* #region reference impl */

pub fn blas_inv_f<T, B>(a: TensorReference<T, B, Ix2>) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat + Send + Sync,
    B: GETRFDriverAPI<T>
        + GETRIDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceComplexFloatAPI<T, Ix2>
        + BlasThreadAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_num_threads();
    device.with_num_threads(nthreads, || {
        let (mut a, ipiv) = GETRF::default().a(a).build()?.run()?;
        GETRI::default().a(a.view_mut()).ipiv(ipiv).build()?.run()?;
        Ok(a)
    })
}

/* #endregion */
