pub mod rstsr_traits {
    pub use rstsr_core::prelude::rstsr_traits::*;
    #[cfg(feature = "linalg")]
    pub mod linalg {
        pub use rstsr_linalg_traits::prelude::rstsr_traits::*;
    }

    #[cfg(feature = "sci")]
    pub mod sci {
        pub use rstsr_sci_traits::prelude::rstsr_traits::*;
    }
}

pub mod rstsr_structs {
    pub use rstsr_core::prelude::rstsr_structs::*;
    #[cfg(feature = "linalg")]
    pub mod linalg {
        pub use rstsr_linalg_traits::prelude::rstsr_structs::*;
    }

    #[cfg(feature = "sci")]
    pub mod sci {
        pub use rstsr_sci_traits::prelude::rstsr_structs::*;
    }

    #[cfg(feature = "aocl")]
    pub use rstsr_aocl::DeviceAOCL;
    #[cfg(feature = "blis")]
    pub use rstsr_blis::DeviceBLIS;
    #[cfg(feature = "mkl")]
    pub use rstsr_mkl::DeviceMKL;
    #[cfg(feature = "openblas")]
    pub use rstsr_openblas::DeviceOpenBLAS;

    #[cfg(all(
        feature = "openblas",
        not(feature = "mkl"),
        not(feature = "blis"),
        not(feature = "aocl"),
        not(feature = "kml")
    ))]
    pub type DeviceBLAS = DeviceOpenBLAS;
    #[cfg(all(
        not(feature = "openblas"),
        feature = "mkl",
        not(feature = "blis"),
        not(feature = "aocl"),
        not(feature = "kml")
    ))]
    pub type DeviceBLAS = DeviceMKL;
    #[cfg(all(
        not(feature = "openblas"),
        not(feature = "mkl"),
        feature = "blis",
        not(feature = "aocl"),
        not(feature = "kml")
    ))]
    pub type DeviceBLAS = DeviceBLIS;
    #[cfg(all(
        not(feature = "openblas"),
        not(feature = "mkl"),
        not(feature = "blis"),
        feature = "aocl",
        not(feature = "kml")
    ))]
    pub type DeviceBLAS = DeviceAOCL;
    #[cfg(all(
        not(feature = "openblas"),
        not(feature = "mkl"),
        not(feature = "blis"),
        not(feature = "aocl"),
        feature = "kml"
    ))]
    pub type DeviceBLAS = rstsr_kml::DeviceKML;
}

pub mod rstsr_funcs {
    pub use rstsr_core::prelude::rstsr_funcs::*;
    #[cfg(feature = "linalg")]
    pub mod linalg {
        pub use rstsr_linalg_traits::prelude::rstsr_funcs::*;
    }

    #[cfg(feature = "sci")]
    pub mod sci {
        pub use rstsr_sci_traits::prelude::rstsr_funcs::*;
    }
}

pub mod rstsr_macros {
    pub use rstsr_core::prelude::rstsr_macros::*;
}

// final re-exports
pub use rstsr_macros::*;
pub use rstsr_structs::*;
pub use rstsr_traits::*;

#[cfg(feature = "linalg")]
pub mod linalg {
    pub use rstsr_linalg_traits::prelude::rstsr_funcs::*;
    pub use rstsr_linalg_traits::prelude::rstsr_structs::*;
    pub use rstsr_linalg_traits::prelude::rstsr_traits::*;
}

#[cfg(feature = "sci")]
pub mod sci {
    pub use rstsr_sci_traits::prelude::rstsr_funcs::*;
    pub use rstsr_sci_traits::prelude::rstsr_mods::*;
    pub use rstsr_sci_traits::prelude::rstsr_structs::*;
    pub use rstsr_sci_traits::prelude::rstsr_traits::*;
}

pub mod rt {
    pub use super::rstsr_funcs;
    pub use super::rstsr_macros;
    pub use super::rstsr_structs;
    pub use super::rstsr_traits;

    pub use super::rstsr_funcs::*;
    pub use super::rstsr_macros::*;
    pub use super::rstsr_structs::*;
    pub use super::rstsr_traits::*;

    #[cfg(feature = "linalg")]
    pub use super::linalg;

    #[cfg(feature = "sci")]
    pub use super::sci;

    pub use rstsr_core::prelude::rt::{Error, Result};
}
