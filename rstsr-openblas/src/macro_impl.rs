pub mod op_binary_arithmetic {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_binary_arithmetic!(DeviceBLAS);
}

pub mod op_binary_common {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_binary_common!(DeviceBLAS);
}

pub mod op_ternary_arithmetic {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_ternary_arithmetic!(DeviceBLAS);
}

pub mod op_ternary_common {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_ternary_common!(DeviceBLAS);
}

pub mod op_tri {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_tri!(DeviceBLAS);
}

pub mod op_with_func {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_op_with_func!(DeviceBLAS);
}

pub mod assignment {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_assignment!(DeviceBLAS);
}

pub mod reduction {
    use crate::DeviceBLAS;
    rstsr_core::macro_impl_rayon_reduction!(DeviceBLAS);
}
