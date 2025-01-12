pub mod op_binary_arithmetic {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_binary_arithmetic!(DeviceOpenBLAS);
}

pub mod op_binary_common {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_binary_common!(DeviceOpenBLAS);
}

pub mod op_ternary_arithmetic {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_ternary_arithmetic!(DeviceOpenBLAS);
}

pub mod op_ternary_common {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_ternary_common!(DeviceOpenBLAS);
}

pub mod op_tri {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_tri!(DeviceOpenBLAS);
}

pub mod op_with_func {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_op_with_func!(DeviceOpenBLAS);
}

pub mod assignment {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_assignment!(DeviceOpenBLAS);
}

pub mod reduction {
    use crate::DeviceOpenBLAS;
    rstsr_core::macro_impl_rayon_reduction!(DeviceOpenBLAS);
}
