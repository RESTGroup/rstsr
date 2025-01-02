use crate::prelude_dev::*;

macro_rules! trait_op_assign_api {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TA: $Op<TB>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn op_muta_refb(
                &self,
                a: &mut Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;

            fn op_muta_numb(&self, a: &mut Storage<TA, Self>, la: &Layout<D>, b: TB) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_assign {
    use super::*;
    use core::ops::*;
    trait_op_assign_api!(DeviceAddAssignAPI   , AddAssign   );
    trait_op_assign_api!(DeviceSubAssignAPI   , SubAssign   );
    trait_op_assign_api!(DeviceMulAssignAPI   , MulAssign   );
    trait_op_assign_api!(DeviceDivAssignAPI   , DivAssign   );
    trait_op_assign_api!(DeviceRemAssignAPI   , RemAssign   );
    trait_op_assign_api!(DeviceBitOrAssignAPI , BitOrAssign );
    trait_op_assign_api!(DeviceBitAndAssignAPI, BitAndAssign);
    trait_op_assign_api!(DeviceBitXorAssignAPI, BitXorAssign);
    trait_op_assign_api!(DeviceShlAssignAPI   , ShlAssign   );
    trait_op_assign_api!(DeviceShrAssignAPI   , ShrAssign   );
}
pub use impl_op_muta_refb_assign::*;

macro_rules! trait_op_l_consume_api {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TA: $Op<TB, Output = TA>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn op_muta_refb(
                &self,
                a: &mut Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>;

            fn op_muta_numb(&self, a: &mut Storage<TA, Self>, la: &Layout<D>, b: TB) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_l_consume {
    use super::*;
    use core::ops::*;
    trait_op_l_consume_api!(DeviceLConsumeAddAPI   , Add   );
    trait_op_l_consume_api!(DeviceLConsumeSubAPI   , Sub   );
    trait_op_l_consume_api!(DeviceLConsumeMulAPI   , Mul   );
    trait_op_l_consume_api!(DeviceLConsumeDivAPI   , Div   );
    trait_op_l_consume_api!(DeviceLConsumeRemAPI   , Rem   );
    trait_op_l_consume_api!(DeviceLConsumeBitOrAPI , BitOr );
    trait_op_l_consume_api!(DeviceLConsumeBitAndAPI, BitAnd);
    trait_op_l_consume_api!(DeviceLConsumeBitXorAPI, BitXor);
    trait_op_l_consume_api!(DeviceLConsumeShlAPI   , Shl   );
    trait_op_l_consume_api!(DeviceLConsumeShrAPI   , Shr   );
}
pub use impl_op_muta_refb_l_consume::*;

macro_rules! trait_op_r_consume_api {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            TA: $Op<TB, Output = TB>,
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn op_muta_refb(
                &self,
                b: &mut Storage<TB, Self>,
                lb: &Layout<D>,
                a: &Storage<TA, Self>,
                la: &Layout<D>,
            ) -> Result<()>;

            fn op_muta_numb(&self, b: &mut Storage<TB, Self>, lb: &Layout<D>, a: TA) -> Result<()>;
        }
    };
}

#[rustfmt::skip]
mod impl_op_muta_refb_r_consume {
    use super::*;
    use core::ops::*;
    trait_op_r_consume_api!(DeviceRConsumeAddAPI   , Add   );
    trait_op_r_consume_api!(DeviceRConsumeSubAPI   , Sub   );
    trait_op_r_consume_api!(DeviceRConsumeMulAPI   , Mul   );
    trait_op_r_consume_api!(DeviceRConsumeDivAPI   , Div   );
    trait_op_r_consume_api!(DeviceRConsumeRemAPI   , Rem   );
    trait_op_r_consume_api!(DeviceRConsumeBitOrAPI , BitOr );
    trait_op_r_consume_api!(DeviceRConsumeBitAndAPI, BitAnd);
    trait_op_r_consume_api!(DeviceRConsumeBitXorAPI, BitXor);
    trait_op_r_consume_api!(DeviceRConsumeShlAPI   , Shl   );
    trait_op_r_consume_api!(DeviceRConsumeShrAPI   , Shr   );
}
pub use impl_op_muta_refb_r_consume::*;

macro_rules! trait_op_unary_api {
    ($DeviceOpAPI:ident, $Op:ident) => {
        pub trait $DeviceOpAPI<TA, TB, D>
        where
            D: DimAPI,
            Self: DeviceAPI<TA> + DeviceAPI<TB>,
        {
            fn op_muta_refb(
                &self,
                a: &mut Storage<TA, Self>,
                la: &Layout<D>,
                b: &Storage<TB, Self>,
                lb: &Layout<D>,
            ) -> Result<()>
            where
                TB: $Op<Output = TA>;

            fn op_muta(&self, a: &mut Storage<TA, Self>, la: &Layout<D>) -> Result<()>
            where
                TA: $Op<Output = TA>;
        }
    };
}

mod impl_op_unary_api {
    use super::*;
    use core::ops::*;
    trait_op_unary_api!(DeviceNegAPI, Neg);
    trait_op_unary_api!(DeviceNotAPI, Not);
}
pub use impl_op_unary_api::*;
