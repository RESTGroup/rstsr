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

/* #region AxisIndex tuple-type from */

// it seems that this directly implementing arbitary AxesIndex<T> will cause
// conflicting implementation so make a macro for this task

#[macro_export]
macro_rules! impl_from_tuple_to_axes_index {
    ($t: ty) => {
        impl<F1, F2> From<(F1, F2)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2>,
        {
            fn from(value: (F1, F2)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3> From<(F1, F2, F3)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3>,
        {
            fn from(value: (F1, F2, F3)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4> From<(F1, F2, F3, F4)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4>,
        {
            fn from(value: (F1, F2, F3, F4)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5> From<(F1, F2, F3, F4, F5)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5>,
        {
            fn from(value: (F1, F2, F3, F4, F5)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5, F6> From<(F1, F2, F3, F4, F5, F6)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5> + TryFrom<F6>,
        {
            fn from(value: (F1, F2, F3, F4, F5, F6)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7> From<(F1, F2, F3, F4, F5, F6, F7)> for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>,
        {
            fn from(value: (F1, F2, F3, F4, F5, F6, F7)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8> From<(F1, F2, F3, F4, F5, F6, F7, F8)>
            for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>,
        {
            fn from(value: (F1, F2, F3, F4, F5, F6, F7, F8)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9> From<(F1, F2, F3, F4, F5, F6, F7, F8, F9)>
            for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>
                + TryFrom<F9>,
        {
            fn from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                    value.8.try_into().ok().unwrap(),
                ])
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9, F10>
            From<(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)> for AxesIndex<$t>
        where
            $t: TryFrom<F1>
                + TryFrom<F2>
                + TryFrom<F3>
                + TryFrom<F4>
                + TryFrom<F5>
                + TryFrom<F6>
                + TryFrom<F7>
                + TryFrom<F8>
                + TryFrom<F9>
                + TryFrom<F10>,
        {
            fn from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)) -> Self {
                AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                    value.8.try_into().ok().unwrap(),
                    value.9.try_into().ok().unwrap(),
                ])
            }
        }
    };
}

impl_from_tuple_to_axes_index!(isize);
impl_from_tuple_to_axes_index!(usize);

/* #endregion AxisIndex tuple-type from */

/* #endregion */
