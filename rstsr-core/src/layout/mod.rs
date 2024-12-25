pub mod broadcast;
pub mod dim;
pub mod indexer;
pub mod iterator_col_major;
pub mod iterator_row_major;
pub mod layoutbase;
pub mod matmul;
pub mod rearrangement;
pub mod shape;
pub mod slice;
pub mod stride;

pub use broadcast::*;
pub use dim::*;
pub use indexer::*;
pub use iterator_col_major::*;
pub use iterator_row_major::*;
pub use layoutbase::*;
pub use matmul::*;
pub use rearrangement::*;
pub use shape::*;
pub use slice::*;
pub use stride::*;

pub trait DimDevAPI: DimBaseAPI + DimShapeAPI + DimStrideAPI + DimLayoutContigAPI {}

impl<const N: usize> DimDevAPI for Ix<N> {}
impl DimDevAPI for IxD {}

pub trait DimAPI:
    DimDevAPI
    + DimIntoAPI<IxD>
    + DimIntoAPI<Ix0>
    + DimIntoAPI<Ix1>
    + DimIntoAPI<Ix2>
    + DimIntoAPI<Ix3>
    + DimIntoAPI<Ix4>
    + DimIntoAPI<Ix5>
    + DimIntoAPI<Ix6>
    + DimIntoAPI<Ix7>
    + DimIntoAPI<Ix8>
    + DimIntoAPI<Ix9>
    + DimIntoAPI<Self>
{
}

impl<const N: usize> DimAPI for Ix<N> {}
impl DimAPI for IxD {}
