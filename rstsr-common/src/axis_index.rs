use crate::prelude_dev::*;

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

impl<T> TryFrom<Vec<T>> for AxesIndex<T> {
    type Error = Error;

    fn try_from(value: Vec<T>) -> Result<Self> {
        Ok(AxesIndex::Vec(value))
    }
}

impl<T, const N: usize> TryFrom<[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    type Error = Error;

    fn try_from(value: [T; N]) -> Result<Self> {
        Ok(AxesIndex::Vec(value.to_vec()))
    }
}

impl<T> TryFrom<&Vec<T>> for AxesIndex<T>
where
    T: Clone,
{
    type Error = Error;

    fn try_from(value: &Vec<T>) -> Result<Self> {
        Ok(AxesIndex::Vec(value.clone()))
    }
}

impl<T> TryFrom<&[T]> for AxesIndex<T>
where
    T: Clone,
{
    type Error = Error;

    fn try_from(value: &[T]) -> Result<Self> {
        Ok(AxesIndex::Vec(value.to_vec()))
    }
}

impl<T, const N: usize> TryFrom<&[T; N]> for AxesIndex<T>
where
    T: Clone,
{
    type Error = Error;

    fn try_from(value: &[T; N]) -> Result<Self> {
        Ok(AxesIndex::Vec(value.to_vec()))
    }
}

impl TryFrom<()> for AxesIndex<usize> {
    type Error = Error;

    fn try_from(_: ()) -> Result<Self> {
        Ok(AxesIndex::Vec(vec![]))
    }
}

impl TryFrom<()> for AxesIndex<isize> {
    type Error = Error;

    fn try_from(_: ()) -> Result<Self> {
        Ok(AxesIndex::Vec(vec![]))
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

impl_try_from_axes_index!(usize, isize, u32, u64, i32, i64);
impl_try_from_axes_index!(isize, usize, u32, u64, i32, i64);

/* #endregion AxisIndex other-type from */

/* #region AxisIndex tuple-type from */

// it seems that this directly implementing arbitary AxesIndex<T> will cause
// conflicting implementation so make a macro for this task

#[macro_export]
macro_rules! impl_from_tuple_to_axes_index {
    ($t: ty) => {
        impl<F1, F2> TryFrom<(F1, F2)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![value.0.try_into().ok().unwrap(), value.1.try_into().ok().unwrap()]))
            }
        }

        impl<F1, F2, F3> TryFrom<(F1, F2, F3)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4> TryFrom<(F1, F2, F3, F4)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5> TryFrom<(F1, F2, F3, F4, F5)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6> TryFrom<(F1, F2, F3, F4, F5, F6)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5> + TryFrom<F6>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7> TryFrom<(F1, F2, F3, F4, F5, F6, F7)> for AxesIndex<$t>
        where
            $t: TryFrom<F1> + TryFrom<F2> + TryFrom<F3> + TryFrom<F4> + TryFrom<F5> + TryFrom<F6> + TryFrom<F7>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8)> for AxesIndex<$t>
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
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8, F9)> for AxesIndex<$t>
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
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
                    value.0.try_into().ok().unwrap(),
                    value.1.try_into().ok().unwrap(),
                    value.2.try_into().ok().unwrap(),
                    value.3.try_into().ok().unwrap(),
                    value.4.try_into().ok().unwrap(),
                    value.5.try_into().ok().unwrap(),
                    value.6.try_into().ok().unwrap(),
                    value.7.try_into().ok().unwrap(),
                    value.8.try_into().ok().unwrap(),
                ]))
            }
        }

        impl<F1, F2, F3, F4, F5, F6, F7, F8, F9, F10> TryFrom<(F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)>
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
                + TryFrom<F9>
                + TryFrom<F10>,
        {
            type Error = Error;

            fn try_from(value: (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)) -> Result<Self> {
                Ok(AxesIndex::Vec(vec![
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
                ]))
            }
        }
    };
}

impl_from_tuple_to_axes_index!(isize);
impl_from_tuple_to_axes_index!(usize);

/* #endregion AxisIndex tuple-type from */
