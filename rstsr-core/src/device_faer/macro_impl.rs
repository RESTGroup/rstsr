pub mod op_binary_arithmetic {
    crate::macro_impl_rayon_op_binary_arithmetic!(DeviceFaer);
}

pub mod op_binary_common {
    crate::macro_impl_rayon_op_binary_common!(DeviceFaer);
}

pub mod op_ternary_arithmetic {
    crate::macro_impl_rayon_op_ternary_arithmetic!(DeviceFaer);
}

pub mod op_ternary_common {
    crate::macro_impl_rayon_op_ternary_common!(DeviceFaer);
}

pub mod op_tri {
    crate::macro_impl_rayon_op_tri!(DeviceFaer);
}

pub mod op_with_func {
    crate::macro_impl_rayon_op_with_func!(DeviceFaer);
}

pub mod assignment {
    crate::macro_impl_rayon_assignment!(DeviceFaer);
}

pub mod reduction {
    crate::macro_impl_rayon_reduction!(DeviceFaer);
}
