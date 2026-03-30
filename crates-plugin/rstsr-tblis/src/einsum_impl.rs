pub use opt_einsum_path::paths::PathOptimizer;
use opt_einsum_path::typing::SizeLimitType;
use rstsr_core::prelude_dev::*;
use rstsr_core::tensor::tensor_view_list::TensorViewListAPI;
pub use tblis::prelude::*;

/// Perform einsum operation using TBLIS.
///
/// This function requires the user to link `tblis.so` or similar libraries that provides TBLIS
/// symbols.
///
/// # Parameters
///
/// - `subscripts`: einsum subscripts, e.g. `"ij,jk->ik"`.
/// - `operands`: list of input tensors (see [`TblisTensor`] for data structure and
///   [`ToTblisTensor`] for conversion trait).
/// - `optimize`: contraction path optimization strategy (see [`opt_einsum_path::contract_path`],
///   usually set to `true`).
/// - `memory_limit`: memory limit for contraction path optimization (see
///   [`opt_einsum_path::contract_path`], usually set to `None`).
///
/// # Returns
///
/// - `Tensor<T, B>`: The output tensor of the einsum operation.
///
/// # Panics
///
/// - This function will panic if failed. Use [`einsum_f`] for fallible version.
/// - This function allows non-ASCII characters in einsum subscripts, but TBLIS may panic if too
///   many characters are used. It is recommended to use no more than 52 characters in total. It is
///   not allowed to use more than 128 characters.
///
/// # Example
///
/// The following example is to perform contraction:
/// $$
/// G_{pqrs} = \sum_{\mu \nu \kappa \lambda} C_{\mu p} C_{\nu q} E_{\mu \nu \kappa \lambda}
/// C_{\kappa r} C_{\lambda s}
/// $$
/// This tensor contraction is utilized in electronic structure (electronic integral in atomic
/// orbital basis $E_{\mu \nu \kappa \lambda}$ to molecular orbital basis $G_{pqrs}$).
///
/// The following code snippet performs this contraction.
///
/// ```rust
/// // Must declare crate `tblis-src` if you want link tblis dynamically.
/// // You can also call the following code in `build.rs`, instead of using crate `tblis-src`:
/// //     println!("cargo:rustc-link-lib=tblis");
/// extern crate tblis_src;
/// use rstsr::prelude::*;
///
/// let device = DeviceFaer::default(); // any device with rayon support can be used.
/// let (nao, nmo): (usize, usize) = (3, 2);
/// let c = rt::arange(((nao * nmo) as f64, &device)).into_shape((nao, nmo));
/// let e = rt::arange(((nao * nao * nao * nao) as f64, &device)).into_shape((nao, nao, nao, nao));
///
/// let g = rt::tblis::einsum(
///     "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
///     [&c, &c, &e, &c, &c],     // tensors to be contracted
///     true,                     // contraction strategy (see crate opt-einsum-path)
///     None,                     // memory limit (None means no limit, see crate opt-einsum-path)
/// );
/// println!("{:?}", g);
/// ```
pub fn einsum<T, B>(
    subscripts: &str,
    operands: impl TensorViewListAPI<T, B>,
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Tensor<T, B>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    einsum_f(subscripts, operands, optimize, memory_limit).unwrap()
}

/// Perform einsum operation using TBLIS.
///
/// See also [`einsum`].
pub fn einsum_with_output<T, B>(
    subscripts: &str,
    operands: impl TensorViewListAPI<T, B>,
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    output: TensorMut<T, B>,
) -> Result<()>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    einsum_with_output_f(subscripts, operands, optimize, memory_limit, output)
}

pub trait RTToTblisTensorAPI<T>
where
    T: TblisFloatAPI,
{
    fn to_tblis_tensor(&self) -> TblisTensor<T>;
}

impl<R, T, B, D> RTToTblisTensorAPI<T> for &TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    fn to_tblis_tensor(&self) -> TblisTensor<T> {
        let shape = self.shape().as_ref().iter().map(|&d| d as isize).collect::<Vec<_>>();
        let stride = self.stride().as_ref().to_vec();
        let offset = self.offset() as isize;
        let ptr = unsafe { self.raw().as_ptr().offset(offset) };
        TblisTensor::new(ptr as *mut T, &shape, &stride)
    }
}

impl<R, T, B, D> RTToTblisTensorAPI<T> for TensorAny<R, T, B, D>
where
    R: DataAPI<Data = B::Raw>,
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
    D: DimAPI,
{
    fn to_tblis_tensor(&self) -> TblisTensor<T> {
        (&self).to_tblis_tensor()
    }
}

pub fn from_tblis_tensor<T, B>(vec: Vec<T>, tblis_tensor: &TblisTensor<T>, device: B) -> Result<Tensor<T, B>>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    let shape = tblis_tensor.shape.iter().map(|&d| d as usize).collect_vec();
    let stride = tblis_tensor.stride.to_vec();
    let offset = 0;
    let layout = Layout::new(shape, stride, offset)?;
    let storage = Storage::new(vec.into(), device);
    Tensor::new_f(storage, layout)
}

/// Perform einsum operation using TBLIS.
///
/// See also [`einsum`].
pub(crate) fn einsum_with_option_output_f<T, B>(
    subscripts: &str,
    operands: impl TensorViewListAPI<T, B>,
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    output: Option<TensorMut<T, B>>,
) -> Result<Option<Tensor<T, B>>>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    // parameter transformation
    let operands = operands.view_list();
    rstsr_assert!(!operands.is_empty(), InvalidValue, "At least one operand is required.")?;

    // check device consistency
    let device = operands[0].device().clone();
    for t in &operands[1..] {
        rstsr_assert!(t.device().same_device(&device), InvalidValue, "All operands must be on the same device.")?;
    }
    if let Some(ref output) = output {
        rstsr_assert!(
            output.device().same_device(&device),
            InvalidValue,
            "Output tensor must be on the same device as operands."
        )?;
    }

    // prepare tblis tensors
    let tblis_operands = operands.iter().map(|t| t.to_tblis_tensor()).collect_vec();
    let tblis_operands_ref = tblis_operands.iter().collect_vec();
    let row_major = device.default_order() == RowMajor;
    let mut tblis_output = output.map(|t| t.to_tblis_tensor());

    // current number of threads
    let num_threads_to_set = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let num_threads_original = tblis_get_num_threads();
    tblis_set_num_threads(num_threads_to_set);

    // call einsum
    let out = unsafe {
        tblis_einsum_f(subscripts, &tblis_operands_ref, optimize, memory_limit, row_major, tblis_output.as_mut())
            .map_err(|e| rstsr_error!(InvalidValue, "TBLIS einsum failed: {e:?}"))?
    };
    tblis_set_num_threads(num_threads_original);

    if let Some((vec, tsr)) = out {
        from_tblis_tensor(vec, &tsr, device).map(Some)
    } else {
        Ok(None)
    }
}

/// Perform einsum operation using TBLIS.
///
/// See also [`einsum`].
pub fn einsum_f<T, B>(
    subscripts: &str,
    operands: impl TensorViewListAPI<T, B>,
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Result<Tensor<T, B>>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    Ok(einsum_with_option_output_f(subscripts, operands, optimize, memory_limit, None)?.unwrap())
}

/// Perform einsum operation using TBLIS.
///
/// See also [`einsum`].
pub fn einsum_with_output_f<T, B>(
    subscripts: &str,
    operands: impl TensorViewListAPI<T, B>,
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    output: TensorMut<T, B>,
) -> Result<()>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    einsum_with_option_output_f(subscripts, operands, optimize, memory_limit, Some(output)).map(|_| ())
}

#[cfg(test)]
mod tests {
    #[test]
    fn playground() {
        extern crate tblis_src;
        use rstsr::prelude::*;

        let device = DeviceFaer::default(); // any device with rayon support can be used.
        let (nao, nmo): (usize, usize) = (3, 2);
        let c = rt::arange(((nao * nmo) as f64, &device)).into_shape((nao, nmo));
        let e = rt::arange(((nao * nao * nao * nao) as f64, &device)).into_shape((nao, nao, nao, nao));

        let g = rt::tblis::einsum(
            "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
            [&c, &c, &e, &c, &c],     // tensors to be contracted
            true,                     // contraction strategy (see crate opt-einsum-path)
            None,                     // memory limit (None means no limit, see crate opt-einsum-path)
        );
        println!("{:?}", g);
    }
}
