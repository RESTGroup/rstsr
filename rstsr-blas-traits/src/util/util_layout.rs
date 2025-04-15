use rstsr_core::prelude::rstsr_structs::*;

pub fn get_output_order(
    by_first: &[Option<(bool, bool)>],
    by_all: &[(bool, bool)],
    default_order: FlagOrder,
) -> FlagOrder {
    // inputs are in the form of (c_prefer, f_prefer)
    for x in by_first {
        match x {
            Some((true, false)) => return RowMajor,
            Some((false, true)) => return ColMajor,
            _ => continue,
        }
    }

    if by_all.iter().any(|&(c_prefer, _)| c_prefer) {
        RowMajor
    } else if by_all.iter().any(|&(_, f_prefer)| f_prefer) {
        ColMajor
    } else {
        default_order
    }
}
