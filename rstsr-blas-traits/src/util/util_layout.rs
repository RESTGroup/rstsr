use rstsr_core::flags::FlagOrder;

pub fn get_order_row_preferred(
    by_first: &[Option<(bool, bool)>],
    by_all: &[(bool, bool)],
) -> FlagOrder {
    // inputs are in the form of (c_prefer, f_prefer)
    for x in by_first {
        if let &Some((c_prefer, f_prefer)) = x {
            if c_prefer {
                return FlagOrder::C;
            } else if f_prefer {
                return FlagOrder::F;
            }
        }
    }

    if by_all.iter().any(|&(c_prefer, _)| c_prefer) {
        FlagOrder::C
    } else if by_all.iter().any(|&(_, f_prefer)| f_prefer) {
        FlagOrder::F
    } else {
        FlagOrder::default()
    }
}
