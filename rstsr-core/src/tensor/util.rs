use crate::prelude_dev::*;

/* #region AxisIndex */

/// Enum for Axes indexing
pub enum AxesIndex<T> {
    Val(T),
    Vec(Vec<T>),
}

impl<T> AsRef<[T]> for AxesIndex<T> {
    fn as_ref(&self) -> &[T] {
        match self {
            AxesIndex::Val(v) => core::slice::from_ref(v),
            AxesIndex::Vec(v) => v.as_slice(),
        }
    }
}

/* #region AxisIndex self-type from */

impl<T> From<T> for AxesIndex<T> {
    fn from(value: T) -> Self {
        AxesIndex::Val(value)
    }
}

impl<T> From<&T> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &T) -> Self {
        AxesIndex::Val(value.clone())
    }
}

impl<T> From<Vec<T>> for AxesIndex<T> {
    fn from(value: Vec<T>) -> Self {
        AxesIndex::Vec(value)
    }
}

impl<T, const N: usize> From<[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: [T; N]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

impl<T> From<&Vec<T>> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &Vec<T>) -> Self {
        AxesIndex::Vec(value.clone())
    }
}

impl<T> From<&[T]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

impl<T, const N: usize> From<&[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    fn from(value: &[T; N]) -> Self {
        AxesIndex::Vec(value.to_vec())
    }
}

impl From<()> for AxesIndex<usize> {
    fn from(_: ()) -> Self {
        AxesIndex::Vec(vec![])
    }
}

/* #endregion AxisIndex self-type from */

/* #region AxisIndex other-type from */

macro_rules! impl_try_from_axes_index {
    ($t1:ty, $($t2:ty),*) => {
        $(
            impl TryFrom<$t2> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: $t2) -> Result<Self> {
                    Ok(AxesIndex::Val(value.try_into()?))
                }
            }

            impl TryFrom<&$t2> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &$t2) -> Result<Self> {
                    Ok(AxesIndex::Val((*value).try_into()?))
                }
            }

            impl TryFrom<Vec<$t2>> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: Vec<$t2>) -> Result<Self> {
                    let value = value
                        .into_iter()
                        .map(|v| v.try_into().map_err(|_| Error::TryFromIntError(String::new())))
                        .collect::<Result<Vec<$t1>>>()?;
                    Ok(AxesIndex::Vec(value))
                }
            }

            impl<const N: usize> TryFrom<[$t2; N]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: [$t2; N]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl TryFrom<&Vec<$t2>> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &Vec<$t2>) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl TryFrom<&[$t2]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &[$t2]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }

            impl<const N: usize> TryFrom<&[$t2; N]> for AxesIndex<$t1> {
                type Error = Error;

                fn try_from(value: &[$t2; N]) -> Result<Self> {
                    value.to_vec().try_into()
                }
            }
        )*
    };
}

impl_try_from_axes_index!(usize, isize, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);
impl_try_from_axes_index!(isize, usize, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/* #endregion AxisIndex other-type from */

/* #endregion */
