use crate::prelude_dev::*;
use rstsr_sci_traits::integrate::lebedev::*;

impl LebedevRuleAPI for DeviceRayonAutoImpl {
    fn lebedev_rule_f(&self, n: usize) -> Result<LebedevQuad<Self>> {
        let degree = lebedev_order_to_degree(n)
            .map_err(|_| rstsr_error!(InvalidValue, "Invalid Lebedev order {n}"))?;
        let (quads, weights) = lebedev_make_angular_grid(degree)?;
        let ngrids = weights.len();
        let quads = asarray((quads, [ngrids, 3].c(), self));
        let weights = asarray((weights, [ngrids].c(), self));
        Ok(LebedevQuad { quads, weights })
    }
}
