use crate::prelude_dev::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxesPairIndex<T> {
    None,
    Val(T),
    Pair(AxesIndex<T>, AxesIndex<T>),
}

impl<X1, X2, T> TryFrom<(X1, X2)> for AxesPairIndex<T>
where
    X1: TryInto<AxesIndex<T>, Error: Into<Error>>,
    X2: TryInto<AxesIndex<T>, Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: (X1, X2)) -> Result<Self> {
        let axes_a = value.0.try_into().map_err(Into::into)?;
        let axes_b = value.1.try_into().map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes_a, axes_b))
    }
}

#[duplicate_item(IType; [i32]; [isize]; [usize])]
#[allow(clippy::unnecessary_cast)]
impl From<IType> for AxesPairIndex<isize> {
    fn from(n: IType) -> Self {
        AxesPairIndex::Val(n as isize)
    }
}

impl<T> From<Option<T>> for AxesPairIndex<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(val) => AxesPairIndex::Val(val),
            None => AxesPairIndex::None,
        }
    }
}

// Mirrors `Option::None` as the "no explicit axes" default (house convention:
// when `()` is a valid overload, `None` must be too).
impl From<()> for AxesPairIndex<isize> {
    fn from(_: ()) -> Self {
        AxesPairIndex::None
    }
}

/* #region AxesPairIndex from a single axes collection (same axes for both operands) */

// A single `axes` collection is shorthand for the pair `(axes, axes)`: the same
// axes are contracted in both operands. These are concrete container impls (not
// a blanket `TryInto<AxesIndex>`) so they cannot collide with the 2-tuple pair
// overload above -- `(0, 1)` stays "axis 0 of a with axis 1 of b", while
// `[0, 1]` means "axis 0 and axis 1 of each of a and b".

impl<X> TryFrom<Vec<X>> for AxesPairIndex<isize>
where
    AxesIndex<isize>: TryFrom<Vec<X>, Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: Vec<X>) -> Result<Self> {
        let axes = AxesIndex::<isize>::try_from(value).map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes.clone(), axes))
    }
}

impl<X, const N: usize> TryFrom<[X; N]> for AxesPairIndex<isize>
where
    AxesIndex<isize>: TryFrom<[X; N], Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: [X; N]) -> Result<Self> {
        let axes = AxesIndex::<isize>::try_from(value).map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes.clone(), axes))
    }
}

impl<'a, X> TryFrom<&'a Vec<X>> for AxesPairIndex<isize>
where
    AxesIndex<isize>: TryFrom<&'a Vec<X>, Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: &'a Vec<X>) -> Result<Self> {
        let axes = AxesIndex::<isize>::try_from(value).map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes.clone(), axes))
    }
}

impl<'a, X> TryFrom<&'a [X]> for AxesPairIndex<isize>
where
    AxesIndex<isize>: TryFrom<&'a [X], Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: &'a [X]) -> Result<Self> {
        let axes = AxesIndex::<isize>::try_from(value).map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes.clone(), axes))
    }
}

impl<'a, X, const N: usize> TryFrom<&'a [X; N]> for AxesPairIndex<isize>
where
    AxesIndex<isize>: TryFrom<&'a [X; N], Error: Into<Error>>,
{
    type Error = Error;

    fn try_from(value: &'a [X; N]) -> Result<Self> {
        let axes = AxesIndex::<isize>::try_from(value).map_err(Into::into)?;
        Ok(AxesPairIndex::Pair(axes.clone(), axes))
    }
}

/* #endregion */

#[cfg(test)]
mod axes_pair_index_tests {
    use crate::prelude_dev::*;

    #[test]
    fn test_same_axes_overload() {
        // a single `axes` collection is shorthand for the pair `(axes, axes)`
        let expect = AxesPairIndex::Pair(AxesIndex::Vec(vec![0, 1]), AxesIndex::Vec(vec![0, 1]));
        let p: AxesPairIndex<isize> = vec![0, 1].try_into().unwrap();
        assert_eq!(p, expect);
        let p: AxesPairIndex<isize> = [0, 1].try_into().unwrap();
        assert_eq!(p, expect);
        let p: AxesPairIndex<isize> = (&vec![0, 1]).try_into().unwrap();
        assert_eq!(p, expect);
        let p: AxesPairIndex<isize> = (&[0, 1][..]).try_into().unwrap();
        assert_eq!(p, expect);
        let p: AxesPairIndex<isize> = (&[0, 1]).try_into().unwrap();
        assert_eq!(p, expect);

        // element types convert like `AxesIndex` does
        let p: AxesPairIndex<isize> = vec![1usize].try_into().unwrap();
        assert_eq!(p, AxesPairIndex::Pair(AxesIndex::Vec(vec![1]), AxesIndex::Vec(vec![1])));
        let p: AxesPairIndex<isize> = [0i32, 2].try_into().unwrap();
        assert_eq!(p, AxesPairIndex::Pair(AxesIndex::Vec(vec![0, 2]), AxesIndex::Vec(vec![0, 2])));

        // the 2-tuple stays the pair overload: (0, 1) is axis 0 of a, axis 1 of b
        let p: AxesPairIndex<isize> = (0, 1).try_into().unwrap();
        assert_eq!(p, AxesPairIndex::Pair(AxesIndex::Val(0), AxesIndex::Val(1)));
    }
}
