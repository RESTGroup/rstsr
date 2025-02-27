use rstsr_core::flags::TensorOrder;

pub fn get_order_row_preferred(
    by_first: &[Option<(bool, bool)>],
    by_all: &[(bool, bool)],
) -> TensorOrder {
    // inputs are in the form of (c_prefer, f_prefer)
    for x in by_first {
        if let &Some((c_prefer, f_prefer)) = x {
            if c_prefer {
                return TensorOrder::C;
            } else if f_prefer {
                return TensorOrder::F;
            }
        }
    }

    if by_all.iter().any(|&(c_prefer, _)| c_prefer) {
        TensorOrder::C
    } else if by_all.iter().any(|&(_, f_prefer)| f_prefer) {
        TensorOrder::F
    } else {
        TensorOrder::default()
    }
}
