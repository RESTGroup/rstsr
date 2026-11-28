use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// n-dim `qr`: the QR factorization of every matrix slice of the stack (the
/// two matrix axes are the last two shape axes under a row-major device, the
/// first two under a column-major one; the rest is the batch); slices may be
/// rectangular `(M, N)`.
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored from a fresh contiguous buffer; a mutable input is
/// destroyed by the per-slice factorization (QR never writes a result back).
#[allow(clippy::type_complexity)]
pub fn blas_impl_qr_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    mode: QrMode,
) -> Result<(Tensor<T, DeviceBLAS, IxD>, Tensor<T, DeviceBLAS, IxD>)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    let [m, n] = [inner_shape[0], inner_shape[1]];
    let k = Ord::min(m, n);
    let (q_inner, r_inner) = match mode {
        QrMode::Reduced => ([m, k], [k, n]),
        QrMode::Complete => ([m, m], [m, n]),
        _ => rstsr_invalid!(mode)?,
    };

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;
    let mut q: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &q_inner, order), &device))?;
    let mut r: Tensor<T, DeviceBLAS, IxD> = zeros_f((stack_shape(stack.batch_shape(), &r_inner, order), &device))?;
    let q_stack = StackLayout::new(q.layout(), 2, order)?;
    let r_stack = StackLayout::new(r.layout(), 2, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let (q_i, r_i) = ref_impl_qr_simple_f(TensorReference::from(slice), mode)?;
        let mut q_slice = stack_slice_at_mut(q.view_mut(), &q_stack, i)?.into_dim::<Ix2>();
        q_slice.assign_f(q_i)?;
        let mut r_slice = stack_slice_at_mut(r.view_mut(), &r_stack, i)?.into_dim::<Ix2>();
        r_slice.assign_f(r_i)?;
    }
    Ok((q, r))
}

/* #region full-args */

impl<T, D, R> QrAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, QrMode)
where
    R: DataAPI<Data = Vec<T>>,
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QrResult<Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>>;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        let (q, r) = blas_impl_qr_f(a.view().to_dyn().into(), mode)?;
        Ok(QrResult { q: q.into_dim::<D>(), r: r.into_dim::<D>() })
    }
}

#[duplicate_item(
    ImplType   ToRef                                       Tr                               ;
    [T, D] [a.view().into_dim::<IxD>().into()       ] [Tensor<T, DeviceBLAS, D>        ];
    ['a, T, D] [a.to_dyn().into()                      ] [TensorView<'a, T, DeviceBLAS, D> ];
    ['a, T, D] [a.view_mut().into_dim::<IxD>().into()  ] [TensorMut<'a, T, DeviceBLAS, D>  ];
)]
impl<ImplType> QrAPI<DeviceBLAS> for (Tr, QrMode)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = QrResult<Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>>;
    // the owned row never mutates `a`
    #[allow(unused_mut)]
    fn qr_f(self) -> Result<Self::Out> {
        let (mut a, mode) = self;
        let (q, r) = blas_impl_qr_f(ToRef, mode)?;
        Ok(QrResult { q: q.into_dim::<D>(), r: r.into_dim::<D>() })
    }
}

#[duplicate_item(
    ImplType                              Tr                               ;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D> ];
   ['a, T, D                          ] [TensorView<'a, T, DeviceBLAS, D> ];
   [T, D                              ] [Tensor<T, DeviceBLAS, D>        ];
   ['a, T, D                          ] [TensorMut<'a, T, DeviceBLAS, D> ];
)]
impl<ImplType> QrAPI<DeviceBLAS> for (Tr, &str)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, QrMode): QrAPI<DeviceBLAS>,
{
    type Out = <(Tr, QrMode) as QrAPI<DeviceBLAS>>::Out;
    fn qr_f(self) -> Result<Self::Out> {
        let (a, mode) = self;
        QrAPI::<DeviceBLAS>::qr_f((a, QrMode::try_from(mode)?))
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplType                              Tr                               ;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D> ];
   ['a, T, D                          ] [TensorView<'a, T, DeviceBLAS, D> ];
   [T, D                              ] [Tensor<T, DeviceBLAS, D>        ];
   ['a, T, D                          ] [TensorMut<'a, T, DeviceBLAS, D> ];
)]
impl<ImplType> QrAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, QrMode): QrAPI<DeviceBLAS>,
{
    type Out = <(Tr, QrMode) as QrAPI<DeviceBLAS>>::Out;
    fn qr_f(self) -> Result<Self::Out> {
        let a = self;
        QrAPI::<DeviceBLAS>::qr_f((a, QrMode::Reduced))
    }
}

/* #endregion */
