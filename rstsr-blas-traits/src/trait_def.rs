//! Convenient traits for BLAS operations.

use crate::blas2::*;
use crate::blas3::*;
use crate::lapack_eig::*;
use crate::lapack_eigh::*;
use crate::lapack_lstsq::*;
use crate::lapack_qr::*;
use crate::lapack_solve::*;
use crate::lapack_svd::*;
use crate::prelude_dev::*;
use rstsr_core::prelude_dev::*;

pub trait BlasDriverBaseAPI<T>:
    // dtype coverage, `Vec`-backed: T, its real and complex counterparts, the integer and bool types
    DeviceAPI<T, Raw = Vec<T>>
    + DeviceAPI<T::Real, Raw = Vec<T::Real>>
    + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
    + DeviceAPI<blas_int, Raw = Vec<blas_int>>
    + DeviceAPI<usize, Raw = Vec<usize>>
    + DeviceAPI<bool, Raw = Vec<bool>>
    + DeviceCreationAnyAPI<bool>
    + DeviceRawAPI<MaybeUninit<T>, Raw = Vec<MaybeUninit<T>>>
    + DeviceRawAPI<MaybeUninit<T::Real>, Raw = Vec<MaybeUninit<T::Real>>>
    + DeviceRawAPI<MaybeUninit<Complex<T::Real>>, Raw = Vec<MaybeUninit<Complex<T::Real>>>>
    + DeviceRawAPI<MaybeUninit<blas_int>, Raw = Vec<MaybeUninit<blas_int>>>
    + DeviceRawAPI<MaybeUninit<usize>, Raw = Vec<MaybeUninit<usize>>>
    + DeviceRawAPI<MaybeUninit<bool>, Raw = Vec<MaybeUninit<bool>>>
    + BlasThreadAPI
    + DeviceRayonAPI
    // lapacke functionality requirements
    + DeviceComplexFloatAPI<T, Ix2>
    + DeviceNumAPI<blas_int, Ix1>
    + DeviceNumAPI<usize, Ix1>
    + OpAddAssignAPI<blas_int, blas_int, Ix1>
    + OpSubAssignAPI<blas_int, blas_int, Ix1>
    // batched scratch matmul (e.g. pinv batch driver, user generic code)
    + DeviceMatMulAPI<T, T, T, Ix2, Ix2, Ix2>
    + DeviceMatMulAPI<T, T, T, IxD, IxD, IxD>
    // linalg functionality requirements
    + OpDivAssignAPI<T, T::Real, IxD>
    + OpAbsAPI<T, Ix1, TOut = T::Real>
    + OpSignAPI<T, Ix1, TOut = T>
    + OpProdAPI<T, Ix1, TOut = T>
    + OpSumAPI<T::Real, Ix1, TOut = T::Real>
    + OpMaxAPI<T::Real, Ix1, TOut = T::Real>
    + OpLogAPI<T::Real, Ix1, TOut = T::Real>
    // n-dim elementwise and reduction ops of the linalg tier (norm, matrix_rank)
    + OpAbsAPI<T, IxD, TOut = T::Real>
    + OpL2NormAPI<T, IxD, TOut = T::Real>
    + OpSignAPI<T::Real, IxD, TOut = T::Real>
    + OpSumAPI<T::Real, IxD, TOut = T::Real>
    + OpMaxAPI<T::Real, IxD, TOut = T::Real>
    + OpMinAPI<T::Real, IxD, TOut = T::Real>
    + OpPowAPI<T::Real, T::Real, IxD, TOut = T::Real>
    + OpMulAPI<T::Real, T::Real, T::Real, IxD>
    + OpGreaterAPI<T::Real, T::Real, IxD, TOut = bool>
    + OpSumBoolAPI<IxD>
    // zero-size identity synthesis (SVD family, ADR-0009 §6)
    + DeviceCreationNumAPI<T>
    // complex counterpart of a real `T` (eig outputs); equals `T` for a complex `T`
    + DeviceCreationAnyAPI<Complex<T::Real>>
    + DeviceCreationNumAPI<Complex<T::Real>>
    + OpAssignAPI<Complex<T::Real>, Ix1>
    + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
    + OpAssignAPI<Complex<T::Real>, Ix2>
    + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
    // vector (Ix1) assignment: contiguity conversions and slice copies of w, s, tau, x, y
    + OpAssignAPI<T, Ix1>
    + OpAssignArbitaryAPI<T, Ix1, Ix1>
    + OpAssignAPI<T::Real, Ix1>
    + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
    // in-place elementwise mutation of a matrix buffer (e.g. conjugation)
    + Op_MutA_API<T, Ix2, fn(&mut MaybeUninit<T>)>
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
    + GEMVDriverAPI<T>
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
    // lapack_eig
    + GEEVDriverAPI<T>
    + GGEVDriverAPI<T>
    // lapack_qr
    + GEQRFDriverAPI<T>
    + ORGQRDriverAPI<T>
    // lapack_solve
    + POTRFDriverAPI<T>
    + GESVDriverAPI<T>
    + GETRFDriverAPI<T>
    + GETRIDriverAPI<T>
    + SYTRFDriverAPI<T, false>
    + SYTRFDriverAPI<T, true>
    + SYSVDriverAPI<T, false>
    + SYSVDriverAPI<T, true>
    // lapack_lstsq
    + GELSDDriverAPI<T>
    // lapack_svd
    + GESVDDriverAPI<T>
    + GESDDDriverAPI<T>
where
    T: BlasFloat,
{
}
