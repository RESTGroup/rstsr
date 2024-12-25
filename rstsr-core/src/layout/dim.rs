use crate::prelude_dev::*;
use core::ops::IndexMut;

/* #region basic definitions */

/// Fixed size dimension.
pub type Ix<const N: usize> = [usize; N];
pub type Ix0 = Ix<0>;
pub type Ix1 = Ix<1>;
pub type Ix2 = Ix<2>;
pub type Ix3 = Ix<3>;
pub type Ix4 = Ix<4>;
pub type Ix5 = Ix<5>;
pub type Ix6 = Ix<6>;
pub type Ix7 = Ix<7>;
pub type Ix8 = Ix<8>;
pub type Ix9 = Ix<9>;
pub type IxD = Vec<usize>;

/// Dynamic size dimension.
pub type IxDyn = IxD;

pub trait DimBaseAPI:
    AsMut<[usize]>
    + AsRef<[usize]>
    + IndexMut<usize, Output = usize>
    + Debug
    + PartialEq
    + Clone
    + TryFrom<Vec<usize>>
    + Into<Vec<usize>>
    + Send
    + Sync
{
    type Stride: AsMut<[isize]>
        + AsRef<[isize]>
        + IndexMut<usize, Output = isize>
        + Debug
        + PartialEq
        + Clone
        + TryFrom<Vec<isize>>
        + Into<Vec<isize>>;

    /// Number of dimension
    fn ndim(&self) -> usize;

    /// Dynamic or static dimension
    fn const_ndim() -> Option<usize>;

    /// New shape
    fn new_shape(&self) -> Self;

    /// New stride
    fn new_stride(&self) -> Self::Stride;
}

impl<const N: usize> DimBaseAPI for Ix<N> {
    type Stride = [isize; N];

    #[inline]
    fn ndim(&self) -> usize {
        N
    }

    #[inline]
    fn const_ndim() -> Option<usize> {
        Some(N)
    }

    #[inline]
    fn new_shape(&self) -> Self {
        [0; N]
    }

    #[inline]
    fn new_stride(&self) -> Self::Stride {
        [0; N]
    }
}

impl DimBaseAPI for IxD {
    type Stride = Vec<isize>;

    #[inline]
    fn ndim(&self) -> usize {
        self.len()
    }

    #[inline]
    fn const_ndim() -> Option<usize> {
        None
    }

    #[inline]
    fn new_shape(&self) -> Self {
        vec![0; self.len()]
    }

    #[inline]
    fn new_stride(&self) -> Self::Stride {
        vec![0; self.len()]
    }
}

/* #endregion */

/* #region dimension relative eq */

// Trait for defining smaller dimension by one.
#[doc(hidden)]
pub trait DimSmallerOneAPI: DimBaseAPI {
    type SmallerOne: DimBaseAPI;
}

// Trait for defining larger dimension by one.
#[doc(hidden)]
pub trait DimLargerOneAPI: DimBaseAPI {
    type LargerOne: DimBaseAPI;
}

impl DimSmallerOneAPI for IxD {
    type SmallerOne = IxD;
}

impl DimLargerOneAPI for IxD {
    type LargerOne = IxD;
}

macro_rules! impl_dim_smaller_one {
    ($(($N:literal, $N1:literal)),*) => {
        $(
            impl DimSmallerOneAPI for Ix<$N> {
                type SmallerOne = Ix<$N1>;
            }
        )*
    };
}

impl_dim_smaller_one!((1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8));

macro_rules! impl_dim_larger_one {
    ($(($N:literal, $N1:literal)),*) => {
        $(
            impl DimLargerOneAPI for Ix<$N> {
                type LargerOne = Ix<$N1>;
            }
        )*
    };
}

impl_dim_larger_one!((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9));

/// Trait for comparing two dimensions.
///
/// This trait is used to broadcast two tensors.
#[doc(hidden)]
pub trait DimMaxAPI<D2>
where
    D2: DimBaseAPI,
{
    // This type will be used in many cases outside layout module.
    // So we use `DimAPI` instead of `DimBaseAPI`, being convenient for
    // functions outside this module.
    type Max: DimAPI;
}

impl DimMaxAPI<IxD> for IxD {
    type Max = IxD;
}

macro_rules! impl_dim_max_dyn {
    ($($N:literal),*) => {
        $(
            impl DimMaxAPI<IxD> for Ix<$N> {
                type Max = IxD;
            }
        )*
    };
}

impl_dim_max_dyn!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

macro_rules! impl_dim_max {
    ($(($N1:literal, $N2:literal, $N:literal)),*) => {
        $(
            impl DimMaxAPI<Ix<$N1>> for Ix<$N2> {
                type Max = Ix<$N>;
            }
        )*
    };
}

impl_dim_max!(
    (0, 0, 0),
    (0, 1, 1),
    (0, 2, 2),
    (0, 3, 3),
    (0, 4, 4),
    (0, 5, 5),
    (0, 6, 6),
    (0, 7, 7),
    (0, 8, 8),
    (0, 9, 9)
);
impl_dim_max!(
    (1, 0, 1),
    (1, 1, 1),
    (1, 2, 2),
    (1, 3, 3),
    (1, 4, 4),
    (1, 5, 5),
    (1, 6, 6),
    (1, 7, 7),
    (1, 8, 8),
    (1, 9, 9)
);
impl_dim_max!(
    (2, 0, 2),
    (2, 1, 2),
    (2, 2, 2),
    (2, 3, 3),
    (2, 4, 4),
    (2, 5, 5),
    (2, 6, 6),
    (2, 7, 7),
    (2, 8, 8),
    (2, 9, 9)
);
impl_dim_max!(
    (3, 0, 3),
    (3, 1, 3),
    (3, 2, 3),
    (3, 3, 3),
    (3, 4, 4),
    (3, 5, 5),
    (3, 6, 6),
    (3, 7, 7),
    (3, 8, 8),
    (3, 9, 9)
);
impl_dim_max!(
    (4, 0, 4),
    (4, 1, 4),
    (4, 2, 4),
    (4, 3, 4),
    (4, 4, 4),
    (4, 5, 5),
    (4, 6, 6),
    (4, 7, 7),
    (4, 8, 8),
    (4, 9, 9)
);
impl_dim_max!(
    (5, 0, 5),
    (5, 1, 5),
    (5, 2, 5),
    (5, 3, 5),
    (5, 4, 5),
    (5, 5, 5),
    (5, 6, 6),
    (5, 7, 7),
    (5, 8, 8),
    (5, 9, 9)
);
impl_dim_max!(
    (6, 0, 6),
    (6, 1, 6),
    (6, 2, 6),
    (6, 3, 6),
    (6, 4, 6),
    (6, 5, 6),
    (6, 6, 6),
    (6, 7, 7),
    (6, 8, 8),
    (6, 9, 9)
);
impl_dim_max!(
    (7, 0, 7),
    (7, 1, 7),
    (7, 2, 7),
    (7, 3, 7),
    (7, 4, 7),
    (7, 5, 7),
    (7, 6, 7),
    (7, 7, 7),
    (7, 8, 8),
    (7, 9, 9)
);
impl_dim_max!(
    (8, 0, 8),
    (8, 1, 8),
    (8, 2, 8),
    (8, 3, 8),
    (8, 4, 8),
    (8, 5, 8),
    (8, 6, 8),
    (8, 7, 8),
    (8, 8, 8),
    (8, 9, 9)
);
impl_dim_max!(
    (9, 0, 9),
    (9, 1, 9),
    (9, 2, 9),
    (9, 3, 9),
    (9, 4, 9),
    (9, 5, 9),
    (9, 6, 9),
    (9, 7, 9),
    (9, 8, 9),
    (9, 9, 9)
);

/* #endregion */
