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
                    a: &mut <Self as DeviceRawAPI<Self::TOut>>::Raw,
                    la: &Layout<D>,
                    b: &<Self as DeviceRawAPI<T>>::Raw,
                    lb: &Layout<D>,
                ) -> Result<()>;

                fn op_muta(&self, a: &mut <Self as DeviceRawAPI<Self::TOut>>::Raw, la: &Layout<D>) -> Result<()>;
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
    DeviceIsFiniteAPI,
    DeviceIsInfAPI,
    DeviceIsNanAPI,
    DeviceLogAPI,
    DeviceLog1pAPI,
    DeviceLog2API,
    DeviceLog10API,
    DeviceRoundAPI,
    DeviceSignBitAPI,
    DeviceSinAPI,
    DeviceSinhAPI,
    DeviceSquareAPI,
    DeviceSqrtAPI,
    DeviceTanAPI,
    DeviceTanhAPI,
    DeviceTruncAPI
);

trait_definition!(
    DeviceRealAbsAPI,
    DeviceRealImagAPI,
    DeviceRealRealAPI,
    DeviceRealSignAPI,
    DeviceComplexAbsAPI,
    DeviceComplexImagAPI,
    DeviceComplexRealAPI,
    DeviceComplexSignAPI
);
