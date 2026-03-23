pub use opt_einsum_path::paths::PathOptimizer;
use rstsr_core::prelude_dev::*;
pub use tblis::prelude::*;

use crate::einsum_impl::einsum_with_option_output_f;

pub fn tensordot<T, B>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    b: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Tensor<T, B>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    tensordot_f(a, b, axes).unwrap()
}

pub fn tensordot_with_output<T, B>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    b: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    output: TensorMut<T, B>,
) -> Result<()>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    tensordot_with_option_output_f(a, b, axes, Some(output)).map(|_| ())
}

pub(crate) fn tensordot_with_option_output_f<T, B>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    b: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    output: Option<TensorMut<T, B>>,
) -> Result<Option<Tensor<T, B>>>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    let a_view = a.view().into_dim::<IxD>();
    let b_view = b.view().into_dim::<IxD>();
    let subscripts = tensordot_to_einsum_str(a_view.ndim(), b_view.ndim(), axes)?;
    einsum_with_option_output_f(&subscripts, (a_view, b_view), true, None, output)
}

pub fn tensordot_with_output_f<T, B>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    b: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
    output: TensorMut<T, B>,
) -> Result<()>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    tensordot_with_option_output_f(a, b, axes, Some(output)).map(|_| ())
}

pub fn tensordot_f<T, B>(
    a: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    b: impl TensorViewAPI<Type = T, Backend = B, Dim: DimAPI>,
    axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>,
) -> Result<Tensor<T, B>>
where
    T: TblisFloatAPI,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceRayonAPI,
{
    Ok(tensordot_with_option_output_f(a, b, axes, None)?.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstsr::prelude::*;

    #[test]
    fn test_tensordot() {
        let device = DeviceFaer::default();

        // a_0 = np.array([[1, 2], [3, 4]])
        // b_0 = np.array([[5, 6], [7, 8]])
        // c_0 = np.tensordot(a_0, b_0, axes=0)
        let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = rt::tensor_from_nested!([[5.0, 6.0], [7.0, 8.0]], &device);
        let result = tensordot(a, b, 0);
        println!("Result of tensordot: \n{:}", result);
        let target = rt::tensor_from_nested!(
            [[[[5, 6], [7, 8]], [[10, 12], [14, 16]]], [[[15, 18], [21, 24]], [[20, 24], [28, 32]]]],
            &device
        );
        assert!(rt::allclose(result, target, None));

        // a = np.arange(60.).reshape(3,4,5)
        // b = np.arange(24.).reshape(4,3,2)
        // c = np.tensordot(a,b, axes=([1,0],[0,1]))
        let a = arange((60.0, &device)).into_shape((3, 4, 5));
        let b = arange((24.0, &device)).into_shape((4, 3, 2));
        let result = tensordot(a, b, ([1, 0], [0, 1]));
        println!("Result of tensordot: \n{:}", result);
        let target = rt::tensor_from_nested!(
            [[4400., 4730.], [4532., 4874.], [4664., 5018.], [4796., 5162.], [4928., 5306.]],
            &device
        );
        assert!(rt::allclose(result, target, None));
    }
}
