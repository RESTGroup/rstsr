#[cfg(feature = "std")]
extern crate std;

use crate::prelude_dev::*;
use core::convert::Infallible;
use core::num::TryFromIntError;
use derive_builder::UninitializedFieldError;
use std::alloc::LayoutError;
use std::collections::TryReserveError;

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    ValueOutOfRange(String),
    InvalidValue(String),
    InvalidLayout(String),
    RuntimeError(String),
    DeviceMismatch(String),
    UnImplemented(String),
    MemoryError(String),

    TryFromIntError(String),
    Infallible,

    BuilderError(UninitializedFieldError),
    DeviceError(String),
    RayonError(String),

    ErrorCode(i32, String),

    Miscellaneous(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<E> = core::result::Result<E, Error>;

impl From<TryFromIntError> for Error {
    fn from(e: TryFromIntError) -> Self {
        Error::TryFromIntError(format!("{e:?}"))
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        Error::Infallible
    }
}

#[cfg(feature = "rayon")]
impl From<rayon::ThreadPoolBuildError> for Error {
    fn from(e: rayon::ThreadPoolBuildError) -> Self {
        Error::RayonError(format!("{e:?}"))
    }
}

impl From<UninitializedFieldError> for Error {
    fn from(e: UninitializedFieldError) -> Self {
        Error::BuilderError(e)
    }
}

impl From<TryReserveError> for Error {
    fn from(e: TryReserveError) -> Self {
        Error::MemoryError(format!("{e:?}"))
    }
}

impl From<LayoutError> for Error {
    fn from(e: LayoutError) -> Self {
        Error::MemoryError(format!("{e:?}"))
    }
}

#[macro_export]
macro_rules! rstsr_assert {
    ($cond:expr, $errtype:ident) => {
        if $cond {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : {:}", stringify!($cond)).unwrap();
            Err(Error::$errtype(s))
        }
    };
    ($cond:expr, $errtype:ident, $($arg:tt)*) => {{
        if $cond {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : ").unwrap();
            write!(s, $($arg)*).unwrap();
            write!(s, " : {:}", stringify!($cond)).unwrap();
            Err(Error::$errtype(s))
        }
    }};
}

#[macro_export]
macro_rules! rstsr_assert_eq {
    ($lhs:expr, $rhs:expr, $errtype:ident) => {
        if $lhs == $rhs {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(
                s,
                " : {:} = {:?} not equal to {:} = {:?}",
                stringify!($lhs),
                $lhs,
                stringify!($rhs),
                $rhs
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
    ($lhs:expr, $rhs:expr, $errtype:ident, $($arg:tt)*) => {
        if $lhs == $rhs {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : ").unwrap();
            write!(s, $($arg)*).unwrap();
            write!(
                s,
                " : {:} = {:?} not equal to {:} = {:?}",
                stringify!($lhs),
                $lhs,
                stringify!($rhs),
                $rhs
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
}

#[macro_export]
macro_rules! rstsr_invalid {
    ($word:expr) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::InvalidValue").unwrap();
        write!(s, " : {:?} = {:?}", stringify!($word), $word).unwrap();
        Err(Error::InvalidValue(s))
    }};
    ($word:expr, $($arg:tt)*) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::InvalidValue").unwrap();
        write!(s, " : {:?} = {:?}", stringify!($word), $word).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error::InvalidValue(s))
    }};
}

#[macro_export]
macro_rules! rstsr_errcode {
    ($word:expr) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::ErrorCode").unwrap();
        write!(s, " : {:?}", $word).unwrap();
        Err(Error::ErrorCode($word, s))
    }};
    ($word:expr, $($arg:tt)*) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::ErrorCode").unwrap();
        write!(s, " : {:?}", $word).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error::ErrorCode($word, s))
    }};
}

#[macro_export]
macro_rules! rstsr_error {
    ($errtype:ident) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        Error::$errtype(s)
    }};
    ($errtype:ident, $($arg:tt)*) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Error::$errtype(s)
    }};
}

#[macro_export]
macro_rules! rstsr_raise {
    ($errtype:ident) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        Err(Error::$errtype(s))
    }};
    ($errtype:ident, $($arg:tt)*) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error::$errtype(s))
    }};
}

#[macro_export]
macro_rules! rstsr_pattern {
    ($value:expr, $pattern:expr, $errtype:ident) => {
        if ($pattern).contains(&($value)) {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(
                s,
                " : {:?} = {:?} not match to pattern {:} = {:?}",
                stringify!($value),
                $value,
                stringify!($pattern),
                $pattern
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
    ($value:expr, $pattern:expr, $errtype:ident, $($arg:tt)*) => {
        if ($pattern).contains(&($value)) {
            Ok(())
        } else {
            use $crate::prelude_dev::*;
            let mut s = String::new();
            write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
            write!(s, concat!("Error::", stringify!($errtype))).unwrap();
            write!(s, " : ").unwrap();
            write!(s, $($arg)*).unwrap();
            write!(
                s,
                " : {:?} = {:?} not match to pattern {:} = {:?}",
                stringify!($value),
                $value,
                stringify!($pattern),
                $pattern
            )
            .unwrap();
            Err(Error::$errtype(s))
        }
    };
}
