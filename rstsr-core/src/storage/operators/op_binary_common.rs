use crate::prelude_dev::*;

macro_rules! trait_definition {
    ($($DeviceOpAPI:ident),*) => {
        $(
            pub trait $DeviceOpAPI<T, D>
            where
                D: DimAPI,
                Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
            {
                type TOut;

                fn op_muta_refb(
                    &self,
                    a: &mut Storage<Self::TOut, Self>,
                    la: &Layout<D>,
                    b: &Storage<T, Self>,
                    lb: &Layout<D>,
                ) -> Result<()>;

                fn op_muta(&self, a: &mut Storage<Self::TOut, Self>, la: &Layout<D>) -> Result<()>;
            }
        )*
    };
}

// Python Array API specifications (2023.1)

// not implemented types
// DeviceBitwiseInvertAPI, (implemented in Not)
// DeviceLogicalNotAPI, (implemented in Not)
// DeviceNegativeAPI, (implemented in Neg)
// DevicePositiveAPI, (not implemented)

trait_definition!(
    DeviceAbsAPI,
    DeviceAcosAPI,
    DeviceAcoshAPI,
    DeviceAsinAPI,
    DeviceAsinhAPI,
    DeviceAtanAPI,
    DeviceAtanhAPI,
    DeviceCeilAPI,
    DeviceConjAPI,
    DeviceCosAPI,
    DeviceCoshAPI,
    DeviceExpAPI,
    DeviceExpm1API,
    DeviceFloorAPI,
    DeviceInvAPI,
    DeviceLogAPI,
    DeviceLog1pAPI,
    DeviceLog2API,
    DeviceLog10API,
    DeviceRoundAPI,
    DeviceSignAPI,
    DeviceSignBitAPI,
    DeviceSinAPI,
    DeviceSinhAPI,
    DeviceSquareAPI,
    DeviceSqrtAPI,
    DeviceTanAPI,
    DeviceTanhAPI,
    DeviceTruncAPI
);

trait_definition!(DeviceImagAPI, DeviceIsFiniteAPI, DeviceIsInfAPI, DeviceIsNanAPI, DeviceRealAPI);
