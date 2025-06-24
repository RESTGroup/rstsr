use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait GESVDDriverAPI<T>
where
    T: BlasFloat,
{
    unsafe fn driver_gesvd(
        order: FlagOrder,
        jobu: char,
        jobvt: char,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        s: *mut T::Real,
        u: *mut T,
        ldu: usize,
        vt: *mut T,
        ldvt: usize,
        superb: *mut T::Real,
    ) -> blas_int;
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct GESVD_<'a, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(default = "true")]
    pub full_matrices: bool,
    #[builder(default = "true")]
    pub compute_uv: bool,
}

impl<'a, B, T> GESVD_<'a, B, T>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + GESVDDriverAPI<T>,
{
    pub fn internal_run(
        self,
    ) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, Option<Tensor<T, B, Ix2>>, Tensor<T::Real, B, Ix1>)>
    {
        let Self { a, full_matrices, compute_uv } = self;

        let device = a.device().clone();
        let order = match (a.c_prefer(), a.f_prefer()) {
            (true, false) => ColMajor,
            (false, true) => RowMajor,
            (false, false) | (true, true) => a.device().default_order(),
        };
        let mut a = overwritable_convert_with_order(a, order)?;
        let [m, n] = *a.view().shape();
        rstsr_assert_eq!(a.view().shape(), &[m, n], InvalidLayout, "GESVD: A shape")?;
        let lda = a.view().ld(order).unwrap();

        // determine job type and matrix sizes
        let jobz = match (compute_uv, full_matrices) {
            (false, _) => 'N',
            (true, false) => 'S',
            (true, true) => 'A',
        };
        let minmn = m.min(n);

        let ([u0, u1], [vt0, vt1]) = match jobz {
            'N' => ([1, 1], [1, 1]),
            'S' => ([m, minmn], [minmn, n]),
            'A' => ([m, m], [n, n]),
            _ => unreachable!(),
        };

        let mut u = unsafe { empty_f(([u0, u1], order, &device))?.into_dim::<Ix2>() };
        let mut vt = unsafe { empty_f(([vt0, vt1], order, &device))?.into_dim::<Ix2>() };
        let mut s = unsafe { empty_f(([minmn], &device))?.into_dim::<Ix1>() };
        let mut superb = unsafe { empty_f(([minmn - 1], &device))?.into_dim::<Ix1>() };

        let ldu = u.view().ld(order).unwrap();
        let ldvt = vt.view().ld(order).unwrap();

        // run driver
        let info = unsafe {
            B::driver_gesvd(
                order,
                jobz,
                jobz,
                m,
                n,
                a.view_mut().as_mut_ptr(),
                lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                ldu,
                vt.as_mut_ptr(),
                ldvt,
                superb.as_mut_ptr(),
            )
        };
        let info = info as i32;
        if info != 0 {
            rstsr_errcode!(info, "Lapack GESVD")?;
        }

        match compute_uv {
            false => Ok((s, None, None, superb)),
            true => Ok((s, Some(u), Some(vt), superb)),
        }
    }

    pub fn run(
        self,
    ) -> Result<(Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>, Option<Tensor<T, B, Ix2>>, Tensor<T::Real, B, Ix1>)>
    {
        self.internal_run()
    }
}

pub type GESVD<'a, B, T> = GESVD_Builder<'a, B, T>;
pub type SGESVD<'a, B> = GESVD<'a, B, f32>;
pub type DGESVD<'a, B> = GESVD<'a, B, f64>;
pub type CGESVD<'a, B> = GESVD<'a, B, Complex<f32>>;
pub type ZGESVD<'a, B> = GESVD<'a, B, Complex<f64>>;
