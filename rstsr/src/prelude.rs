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

    #[cfg(feature = "openblas")]
    pub use rstsr_openblas::DeviceOpenBLAS;
    #[cfg(feature = "openblas")]
    pub type DeviceBLAS = DeviceOpenBLAS;
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
