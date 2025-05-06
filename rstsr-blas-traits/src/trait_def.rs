//! Convenient traits for BLAS operations.

use crate::blas3::*;
use crate::lapack_eigh::*;
use crate::lapack_solve::*;
use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait BlasDriverBaseAPI<T>:
    DeviceAPI<T, Raw = Vec<T>>
    + DeviceAPI<T::Real, Raw = Vec<T::Real>>
    + DeviceAPI<blas_int, Raw = Vec<blas_int>>
    + DeviceComplexFloatAPI<T, Ix2>
    + DeviceNumAPI<blas_int, Ix1>
    + BlasThreadAPI
    + DeviceRayonAPI
where
    T: BlasFloat,
{
}

pub trait BlasDriverAPI<T>:
    BlasDriverBaseAPI<T>
    + GEMMDriverAPI<T>
    + TRSMDriverAPI<T>
    + SYHEMMDriverAPI<T, false>
    + SYHEMMDriverAPI<T, true>
where
    T: BlasFloat,
{
}

pub trait LapackDriverAPI<T>:
    BlasDriverBaseAPI<T>
    + BlasDriverAPI<T>
    // lapack_eigh
    + SYEVDriverAPI<T>
    + SYEVDDriverAPI<T>
    + SYGVDriverAPI<T>
    + SYGVDDriverAPI<T>
    // lapack_solve
    + POTRFDriverAPI<T>
    + GESVDriverAPI<T>
    + GETRFDriverAPI<T>
    + GETRIDriverAPI<T>
    + SYSVDriverAPI<T, false>
    + SYSVDriverAPI<T, true>
where
    T: BlasFloat,
{
}
