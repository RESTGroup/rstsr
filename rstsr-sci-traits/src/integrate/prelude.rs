pub mod rstsr_traits {
    pub use crate::integrate::lebedev::LebedevRuleAPI;
}

pub mod rstsr_funcs {
    pub use crate::integrate::lebedev::{
        lebedev_rule, lebedev_rule_f, lebedev_rule_from_degree, lebedev_rule_from_degree_f,
    };
}

pub mod rstsr_structs {
    pub use crate::integrate::lebedev::LebedevQuad;
}
