pub mod cblas;

#[cfg(not(feature = "lapacke"))]
pub mod lapack;
#[cfg(feature = "lapacke")]
pub mod lapacke;
