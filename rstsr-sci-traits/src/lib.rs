// Self-alias for files symlinked across crates: `use rstsr_sci_traits::...`
// resolves to `self` here and to the dependency downstream. An explicit
// `extern crate self` (not a glob re-export in `prelude_dev`) avoids E0659
// under rustdoc doctest compilation.
extern crate self as rstsr_sci_traits;

pub mod prelude;
pub mod prelude_dev;

pub mod distance;
pub mod integrate;
