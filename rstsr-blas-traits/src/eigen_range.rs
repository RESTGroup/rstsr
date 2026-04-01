use crate::prelude_dev::*;
use num::Zero;

/// Range selection for eigenvalue subset.
#[derive(Clone, Copy, Debug, Default)]
pub enum EigenRange<T: BlasFloat> {
    /// Select all eigenvalues (range = 'A')
    #[default]
    All,
    /// Select eigenvalues in the interval (vl, vu] (range = 'V')
    Value(T::Real, T::Real),
    /// Select eigenvalues with indices il through iu (0-indexed, range = 'I').
    /// If iu is None, selects from il to the end.
    Index(usize, Option<usize>),
}

// From implementations for EigenRange

impl<T: BlasFloat> From<(T::Real, T::Real)> for EigenRange<T> {
    fn from((lo, hi): (T::Real, T::Real)) -> Self {
        EigenRange::Value(lo, hi)
    }
}

impl<T: BlasFloat> From<std::ops::Range<usize>> for EigenRange<T> {
    fn from(range: std::ops::Range<usize>) -> Self {
        EigenRange::Index(range.start, Some(range.end.saturating_sub(1)))
    }
}

impl<T: BlasFloat> From<std::ops::RangeTo<usize>> for EigenRange<T> {
    fn from(range: std::ops::RangeTo<usize>) -> Self {
        EigenRange::Index(0, Some(range.end.saturating_sub(1)))
    }
}

impl<T: BlasFloat> From<std::ops::RangeFrom<usize>> for EigenRange<T> {
    fn from(range: std::ops::RangeFrom<usize>) -> Self {
        EigenRange::Index(range.start, None)
    }
}

impl<T: BlasFloat> From<std::ops::RangeFull> for EigenRange<T> {
    fn from(_: std::ops::RangeFull) -> Self {
        EigenRange::All
    }
}

impl<T: BlasFloat> From<std::ops::RangeInclusive<usize>> for EigenRange<T> {
    fn from(range: std::ops::RangeInclusive<usize>) -> Self {
        let (start, end) = range.into_inner();
        EigenRange::Index(start, Some(end))
    }
}

impl<T: BlasFloat> From<std::ops::RangeToInclusive<T::Real>> for EigenRange<T> {
    fn from(range: std::ops::RangeToInclusive<T::Real>) -> Self {
        EigenRange::Value(T::Real::zero(), range.end)
    }
}
