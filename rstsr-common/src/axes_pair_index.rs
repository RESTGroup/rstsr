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
