use crate::prelude_dev::*;
use core::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Stride<D>(pub D::Stride)
where
    D: DimBaseAPI;

impl<D> Deref for Stride<D>
where
    D: DimBaseAPI,
{
    type Target = D::Stride;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D> DerefMut for Stride<D>
where
    D: DimBaseAPI,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait DimStrideAPI: DimBaseAPI {
    /// Number of dimensions of the shape.
    fn ndim(stride: &Stride<Self>) -> usize;
}

impl<D> Stride<D>
where
    D: DimStrideAPI,
{
    pub fn ndim(&self) -> usize {
        <D as DimStrideAPI>::ndim(self)
    }
}

impl<const N: usize> DimStrideAPI for Ix<N> {
    fn ndim(stride: &Stride<Ix<N>>) -> usize {
        stride.len()
    }
}

impl DimStrideAPI for IxD {
    fn ndim(stride: &Stride<IxD>) -> usize {
        stride.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ndim() {
        // general test
        let stride = Stride::<Ix2>([2, 3]);
        assert_eq!(stride.ndim(), 2);
        let stride = Stride::<IxD>(vec![2, 3]);
        assert_eq!(stride.ndim(), 2);
        // empty dimension test
        let stride = Stride::<Ix0>([]);
        assert_eq!(stride.ndim(), 0);
        let stride = Stride::<IxD>(vec![]);
        assert_eq!(stride.ndim(), 0);
    }
}
