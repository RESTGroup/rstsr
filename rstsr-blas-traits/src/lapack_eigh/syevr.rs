use crate::eigen_range::EigenRange;
use crate::prelude_dev::*;
use num::Zero;
use rstsr_core::prelude_dev::*;

/// LAPACK SYEVR/HEEVR driver API for symmetric/Hermitian eigenvalue problem with subset selection.
///
/// Computes eigenvalues and optionally eigenvectors of a symmetric/Hermitian matrix
/// with support for selecting a subset of eigenvalues by index or value range.
pub trait SYEVRDriverAPI<T>
where
    T: BlasFloat,
{
    /// LAPACK SYEVR/HEEVR driver function.
    ///
    /// # Safety
    ///
    /// This function calls LAPACK routines directly and modifies memory in place.
    unsafe fn driver_syevr(
        order: FlagOrder,
        jobz: char,
        range: char,
        uplo: FlagUpLo,
        n: usize,
        a: *mut T,
        lda: usize,
        vl: T::Real,
        vu: T::Real,
        il: blas_int,
        iu: blas_int,
        abstol: T::Real,
        m: *mut blas_int,
        w: *mut T::Real,
        z: *mut T,
        ldz: usize,
        isuppz: *mut blas_int,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SYEVR_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,

    #[builder(setter(into), default = "'V'")]
    pub jobz: char,
    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
    #[builder(setter(into), default = "EigenRange::All")]
    pub range: EigenRange<T>,
    #[builder(setter(into), default = "T::Real::zero()")]
    pub abstol: T::Real,
}

impl<'a, B, T> SYEVR_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + SYEVRDriverAPI<T>,
{
    pub fn internal_run(
        self,
    ) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, TensorMutable<'a, T, B, Ix2>)> {
        let Self { a, jobz, uplo, range, abstol } = self;

        let device = a.device().clone();
        let uplo = uplo.unwrap_or_else(|| match device.default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        });
        let mut a = overwritable_convert(a)?;
        let order = if a.f_prefer() && !a.c_prefer() { ColMajor } else { RowMajor };

        let [n, m] = *a.view().shape();
        rstsr_assert_eq!(n, m, InvalidLayout)?;

        let lda = a.view().ld(order).unwrap();

        // Determine range parameters
        let (range_char, vl, vu, il, iu) = match range {
            EigenRange::All => ('A', T::Real::zero(), T::Real::zero(), 0, 0),
            EigenRange::Value(lo, hi) => ('V', lo, hi, 0, 0),
            EigenRange::Index(lo, Some(hi)) => {
                rstsr_assert!(lo <= hi && hi < n, InvalidLayout)?;
                // LAPACK uses 1-indexed values
                ('I', T::Real::zero(), T::Real::zero(), (lo + 1) as blas_int, (hi + 1) as blas_int)
            },
            EigenRange::Index(lo, None) => {
                rstsr_assert!(lo < n, InvalidLayout)?;
                // From lo to the end: iu = n
                ('I', T::Real::zero(), T::Real::zero(), (lo + 1) as blas_int, n as blas_int)
            },
        };

        // Allocate output arrays
        // w is always n elements (LAPACK requirement)
        let mut w = unsafe { empty_f(([n].c(), &device))?.into_dim::<Ix1>() };
        let mut m_val: blas_int = 0;

        // Determine expected number of eigenvalues (ncols_z in LAPACKE terminology)
        let expected_m = match range {
            EigenRange::All => n,
            EigenRange::Value(_, _) => n, // Could be anything up to n
            EigenRange::Index(lo, Some(hi)) => hi - lo + 1,
            EigenRange::Index(lo, None) => n - lo,
        };

        // Allocate z only if eigenvectors requested
        // LAPACKE row-major: Z has logical shape (n, expected_m), ldz >= expected_m
        // LAPACKE col-major: Z has shape (n, expected_m), ldz >= n
        let compute_z = jobz == 'V' || jobz == 'v';
        let (ldz, mut z) = if compute_z {
            if order == RowMajor {
                // Row-major: allocate (n, expected_m), ldz = expected_m
                let ldz = expected_m.max(1);
                let z = unsafe { empty_f(([n, expected_m].c(), &device))?.into_dim::<Ix2>() };
                (ldz, Some(z))
            } else {
                // Column-major: allocate (n, expected_m), ldz = n
                let ldz = n.max(1);
                let z = unsafe { empty_f(([n, expected_m].c(), &device))?.into_dim::<Ix2>() };
                (ldz, Some(z))
            }
        } else {
            (1, None)
        };

        // isuppz is needed when jobz = 'V'
        let mut isuppz =
            if compute_z { Some(unsafe { empty_f(([2 * expected_m].c(), &device))?.into_dim::<Ix1>() }) } else { None };

        let ptr_a = a.view_mut().as_mut_ptr();
        let ptr_w = w.as_mut_ptr();
        let ptr_z = z.as_mut().map(|z| z.view_mut().as_mut_ptr()).unwrap_or(std::ptr::null_mut());
        let ptr_isuppz = isuppz.as_mut().map(|s| s.as_mut_ptr()).unwrap_or(std::ptr::null_mut());

        // run driver
        let info = unsafe {
            B::driver_syevr(
                order, jobz, range_char, uplo, n, ptr_a, lda, vl, vu, il, iu, abstol, &mut m_val, ptr_w, ptr_z, ldz,
                ptr_isuppz,
            )
        };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack SYEVR")?;
        }

        // Get actual number of eigenvalues found
        let m_found = m_val as usize;

        // Slice w to actual number of eigenvalues found
        let w = w.i(..m_found).into_owned().into_dim::<Ix1>();

        // Slice z to actual number of eigenvectors found
        // For both row and column major, Z has shape (n, expected_m)
        // After slicing: (n, m_found)
        let z = z.map(|z_tensor| z_tensor.i((.., ..m_found)).into_owned().into_dim::<Ix2>());

        Ok((w, z, a.clone_to_mut()))
    }

    pub fn run(self) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, TensorMutable<'a, T, B, Ix2>)> {
        self.internal_run()
    }
}

pub type SYEVR<'a, B, T> = SYEVR_Builder<'a, B, T>;
pub type SSYEVR<'a, B> = SYEVR<'a, B, f32>;
pub type DSYEVR<'a, B> = SYEVR<'a, B, f64>;
pub type CHEEVR<'a, B> = SYEVR<'a, B, Complex<f32>>;
pub type ZHEEVR<'a, B> = SYEVR<'a, B, Complex<f64>>;
