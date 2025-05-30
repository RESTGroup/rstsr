pub mod rstsr_traits {
    pub use crate::distance::metric::MetricDistAPI;
    pub use crate::distance::traits::CDistAPI;
}

pub mod rstsr_funcs {
    pub use crate::distance::traits::{cdist, cdist_f};
}

pub mod rstsr_structs {
    pub use crate::distance::metric::MetricEuclidean;
}
