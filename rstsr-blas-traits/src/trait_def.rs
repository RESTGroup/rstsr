//! Convenient traits for BLAS operations.

use crate::blas3::*;
use crate::lapack_eigh::*;
use crate::lapack_solve::*;
use crate::lapack_svd::*;
use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait BlasDriverBaseAPI<T>:
    DeviceAPI<T, Raw = Vec<T>>
    + DeviceAPI<T::Real, Raw = Vec<T::Real>>
    + DeviceAPI<blas_int, Raw = Vec<blas_int>>
    + BlasThreadAPI
    + DeviceRayonAPI
    // lapacke functionality requirements
    + DeviceComplexFloatAPI<T, Ix2>
    + DeviceNumAPI<blas_int, Ix1>
    + DeviceAddAssignAPI<blas_int, blas_int, Ix1>
    + DeviceSubAssignAPI<blas_int, blas_int, Ix1>
    // linalg functionality requirements
    + DeviceDivAssignAPI<T, T::Real, IxD>
    + DeviceAbsAPI<T, Ix1, TOut = T::Real>
    + DeviceSignAPI<T, Ix1, TOut = T>
    + OpProdAPI<T, Ix1, TOut = T>
    + OpSumAPI<T::Real, Ix1, TOut = T::Real>
    + OpMaxAPI<T::Real, Ix1, TOut = T::Real>
    + DeviceLogAPI<T::Real, Ix1, TOut = T::Real>
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
    // lapack_svd
    + GESVDDriverAPI<T>
    + GESDDDriverAPI<T>
where
    T: BlasFloat,
{
}
