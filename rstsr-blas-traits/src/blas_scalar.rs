use half::{bf16, f16};
use num::complex::ComplexFloat;
use num::{Complex, Num};
use rstsr_dtype_traits::{ExtNum, ExtReal};
use std::ffi::c_void;
use std::ops::*;

pub trait BlasScalar: Num + Clone {
    type FFI;
    type Scalar;
}

impl BlasScalar for f32 {
    type FFI = f32;
    type Scalar = f32;
}

impl BlasScalar for f64 {
    type FFI = f64;
    type Scalar = f64;
}

impl BlasScalar for Complex<f32> {
    type FFI = c_void;
    type Scalar = *const c_void;
}

impl BlasScalar for Complex<f64> {
    type FFI = c_void;
    type Scalar = *const c_void;
}

impl BlasScalar for f16 {
    type FFI = f16;
    type Scalar = f16;
}

impl BlasScalar for bf16 {
    type FFI = bf16;
    type Scalar = bf16;
}

pub trait BlasFloat:
    BlasScalar
    + ComplexFloat<Real: ExtReal>
    + Send
    + Sync
    + Div<Self::Real, Output = Self>
    + DivAssign<Self::Real>
    + Mul<Self::Real, Output = Self>
    + ExtNum<AbsOut = Self::Real>
{
}
impl BlasFloat for f32 {}
impl BlasFloat for f64 {}
impl BlasFloat for Complex<f32> {}
impl BlasFloat for Complex<f64> {}
impl BlasFloat for f16 {}
impl BlasFloat for bf16 {}
