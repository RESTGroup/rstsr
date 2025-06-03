pub mod rstsr_traits {
    pub use crate::distance::metric::{MetricDistAPI, MetricDistWeightedAPI};
    pub use crate::distance::traits::CDistAPI;
}

pub mod rstsr_funcs {
    pub use crate::distance::traits::{cdist, cdist_f};
}

pub mod rstsr_structs {
    pub use crate::distance::metric::{
        MetricBrayCurtis, MetricCanberra, MetricChebyshev, MetricCityBlock, MetricCorrelation,
        MetricCosine, MetricDice, MetricEuclidean, MetricHamming, MetricJaccard,
        MetricJensenShannon, MetricMinkowski, MetricRogersTanimoto, MetricRussellRao,
        MetricSokalSneath, MetricSqEuclidean, MetricYule,
    };
}
