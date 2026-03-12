use crate::prelude_dev::*;

#[duplicate_item(
    OpAPI           ;
   [OpAcosAPI      ];
   [OpAcoshAPI     ];
   [OpAsinAPI      ];
   [OpAsinhAPI     ];
   [OpAtanAPI      ];
   [OpAtanhAPI     ];
   [OpCeilAPI      ];
   [OpConjAPI      ];
   [OpCosAPI       ];
   [OpCoshAPI      ];
   [OpExpAPI       ];
   [OpExpm1API     ];
   [OpFloorAPI     ];
   [OpInvAPI       ];
   [OpIsFiniteAPI  ];
   [OpIsInfAPI     ];
   [OpIsNanAPI     ];
   [OpLogAPI       ];
   [OpLog1pAPI     ];
   [OpLog2API      ];
   [OpLog10API     ];
   [OpReciprocalAPI];
   [OpRoundAPI     ];
   [OpSignBitAPI   ];
   [OpSinAPI       ];
   [OpSinhAPI      ];
   [OpSquareAPI    ];
   [OpSqrtAPI      ];
   [OpTanAPI       ];
   [OpTanhAPI      ];
   [OpTruncAPI     ];
   [OpAbsAPI       ];
   [OpImagAPI      ];
   [OpRealAPI      ];
   [OpSignAPI      ];
)]
pub trait OpAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<MaybeUninit<Self::TOut>>,
{
    type TOut;

    fn op_muta_refb(
        &self,
        a: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<T>>::Raw,
        lb: &Layout<D>,
    ) -> Result<()>;

    fn op_muta(&self, a: &mut <Self as DeviceRawAPI<MaybeUninit<Self::TOut>>>::Raw, la: &Layout<D>) -> Result<()>;
}

// Python Array API specifications (2023.1)

// not implemented types
// DeviceBitwiseInvertAPI, (implemented in Not)
// DeviceLogicalNotAPI, (implemented in Not)
// DeviceNegativeAPI, (implemented in Neg)
// DevicePositiveAPI, (not implemented)
