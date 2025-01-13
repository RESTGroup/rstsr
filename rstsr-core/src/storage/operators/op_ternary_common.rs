use crate::prelude_dev::*;

macro_rules! trait_definition {
    ($($DeviceOpAPI:ident),*) => {
        $(
            pub trait $DeviceOpAPI<TA, TB, D>
            where
                D: DimAPI,
                Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<Self::TOut>,
            {
                type TOut;

                fn op_mutc_refa_refb(
                    &self,
                    c: &mut <Self as DeviceRawAPI<Self::TOut>>::Raw,
                    lc: &Layout<D>,
                    a: &<Self as DeviceRawAPI<TA>>::Raw,
                    la: &Layout<D>,
                    b: &<Self as DeviceRawAPI<TB>>::Raw,
                    lb: &Layout<D>,
                ) -> Result<()>;
            }
        )*
    };
}

// Python Array API specifications (2023.1)

// not implemented types (defined in arithmetics)
// add, bitwise_and, bitwise_left_shift, bitwise_or, bitwise_right_shift,
// bitwise_xor, divide, logical_and, logical_or, multiply, remainder, subtract

trait_definition!(
    DeviceATan2API,
    DeviceCopySignAPI,
    DeviceEqualAPI,
    DeviceGreaterAPI,
    DeviceGreaterEqualAPI,
    DeviceHypotAPI,
    DeviceLessAPI,
    DeviceLessEqualAPI,
    DeviceLogAddExpAPI,
    DeviceMaximumAPI,
    DeviceMinimumAPI,
    DeviceNotEqualAPI,
    DevicePowAPI
);

trait_definition!(DeviceFloatFloorDivideAPI, DeviceIntFloorDivideAPI);
