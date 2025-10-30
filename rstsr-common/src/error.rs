#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use crate::prelude_dev::*;
use alloc::collections::TryReserveError;
use core::alloc::LayoutError;
use core::convert::Infallible;
use core::num::TryFromIntError;
use derive_builder::UninitializedFieldError;

#[non_exhaustive]
#[derive(Debug)]
pub enum RSTSRError {
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
    FaerError(String),

    Miscellaneous(String),
}

#[derive(Debug)]
pub struct Error {
    pub inner: RSTSRError,
    pub backtrace: Option<String>,
}

pub fn rstsr_backtrace() -> Option<String> {
    #[cfg(feature = "backtrace")]
    {
        extern crate std;
        let bt = std::backtrace::Backtrace::capture();
        Some(format!("{:}", bt))
    }
    #[cfg(not(feature = "backtrace"))]
    {
        None
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

pub trait RSTSRResultAPI<T> {
    fn rstsr_unwrap(self) -> T;
}

impl<T> RSTSRResultAPI<T> for Result<T> {
    #[allow(unused_variables)]
    fn rstsr_unwrap(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => {
                let Error { inner, backtrace } = &e;
                #[cfg(feature = "backtrace")]
                {
                    extern crate std;
                    if let Some(backtrace) = backtrace {
                        std::eprintln!("\n====== RSTSR Backtrace ======\n{:}", backtrace);
                    }
                }
                panic!("RSTSR Error: {:?}", inner)
            },
        }
    }
}

impl From<TryFromIntError> for Error {
    fn from(e: TryFromIntError) -> Self {
        Error { inner: RSTSRError::TryFromIntError(format!("{e:?}")), backtrace: rstsr_backtrace() }
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        Error { inner: RSTSRError::Infallible, backtrace: rstsr_backtrace() }
    }
}

#[cfg(feature = "rayon")]
impl From<rayon::ThreadPoolBuildError> for Error {
    fn from(e: rayon::ThreadPoolBuildError) -> Self {
        Error { inner: RSTSRError::RayonError(format!("{e:?}")), backtrace: rstsr_backtrace() }
    }
}

impl From<UninitializedFieldError> for Error {
    fn from(e: UninitializedFieldError) -> Self {
        Error { inner: RSTSRError::BuilderError(e), backtrace: rstsr_backtrace() }
    }
}

impl From<TryReserveError> for Error {
    fn from(e: TryReserveError) -> Self {
        Error { inner: RSTSRError::MemoryError(format!("{e:?}")), backtrace: rstsr_backtrace() }
    }
}

impl From<LayoutError> for Error {
    fn from(e: LayoutError) -> Self {
        Error { inner: RSTSRError::MemoryError(format!("{e:?}")), backtrace: rstsr_backtrace() }
    }
}

#[macro_export]
macro_rules! backtrace {
    () => {{
        #[cfg(feature = "backtrace")]
        {
            extern crate std;
            let bt = std::backtrace::Backtrace::capture();
            format!("\nBacktrace:\n{:}", bt)
        }
        #[cfg(not(feature = "backtrace"))]
        {
            String::new()
        }
    }};
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
        Err(Error{ inner: RSTSRError::InvalidValue(s), backtrace: rstsr_backtrace() })
    }};
    ($word:expr, $($arg:tt)*) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::InvalidValue").unwrap();
        write!(s, " : {:?} = {:?}", stringify!($word), $word).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error{ inner: RSTSRError::InvalidValue(s), backtrace: rstsr_backtrace() })
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
        Err(Error{ inner: RSTSRError::ErrorCode($word, s), backtrace: rstsr_backtrace() })
    }};
    ($word:expr, $($arg:tt)*) => {{
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, "Error::ErrorCode").unwrap();
        write!(s, " : {:?}", $word).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error{ inner: RSTSRError::ErrorCode($word, s), backtrace: rstsr_backtrace() })
    }};
}

#[macro_export]
macro_rules! rstsr_error {
    ($errtype:ident) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() }
    }};
    ($errtype:ident, $($arg:tt)*) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() }
    }};
}

#[macro_export]
macro_rules! rstsr_raise {
    ($errtype:ident) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
    }};
    ($errtype:ident, $($arg:tt)*) => {{
        use $crate::prelude_dev::*;
        let mut s = String::new();
        write!(s, concat!(file!(), ":", line!(), ": ")).unwrap();
        write!(s, concat!("Error::", stringify!($errtype))).unwrap();
        write!(s, " : ").unwrap();
        write!(s, $($arg)*).unwrap();
        Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
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
            Err(Error{ inner: RSTSRError::$errtype(s), backtrace: rstsr_backtrace() })
        }
    };
}
