use num::{complex::ComplexFloat, Float, One, ToPrimitive, Zero};
use rstsr_core::prelude_dev::*;

/// Distance metric API trait.
///
/// # Generic Arguments
/// * `T` - The type of the distance metric result (e.g., f32, f64).
/// * `V` - The type of the vectors (`Vec<T>` in general, but can be other)
pub trait MetricDistAPI<V> {
    /// The output type of the distance metric.
    type Out;

    /// Computes the distance between two vectors.
    ///
    /// This function only accepts contiguous vectors (not strided).
    ///
    /// # Arguments
    ///
    /// * `uv` - The vectors to compare.
    /// * `offsets` - The offsets in the vectors (starting point).
    /// * `indices` - The indices of the vectors to compare (generally not used,
    ///   but will be applied in cosine metric).
    /// * `strides` - The strides of the vectors (will not be applied if
    ///   `STRIDED` is false).
    /// * `size` - The number of elements to consider in the distance
    ///   calculation.
    fn distance<const STRIDED: bool>(
        &self,
        uv: (&V, &V),
        offsets: (usize, usize),
        indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> Self::Out;
}

/// Distance metric API trait.
///
/// # Generic Arguments
/// * `T` - The type of the distance metric result (e.g., f32, f64).
/// * `V` - The type of the vectors (`Vec<T>` in general, but can be other)
pub trait MetricDistWeightedAPI<V> {
    // The type of the weights, generally same to `V`, but may be `Vec<f64>`
    // sometimes.
    type Weight;

    /// The output type of the distance metric.
    type Out;

    /// Computes the distance between two vectors.
    ///
    /// This function only accepts contiguous vectors (not strided).
    ///
    /// # Arguments
    ///
    /// * `uv` - The vectors to compare.
    /// * `weights` - The weights to apply to the vectors (will not be applied
    ///   if `WEIGHTED` is false).
    /// * `offsets` - The offsets in the vectors (starting point).
    /// * `indices` - The indices of the vectors to compare (generally not used,
    ///   but will be applied in cosine metric).
    /// * `strides` - The strides of the vectors (will not be applied if
    ///   `STRIDED` is false).
    /// * `size` - The number of elements to consider in the distance
    ///   calculation.
    fn weighted_distance<const STRIDED: bool>(
        &self,
        uv: (&V, &V),
        weights: &Self::Weight,
        offsets: (usize, usize),
        indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> Self::Out;
}

/* #region zero-size distance types */

pub struct MetricEuclidean;
pub struct MetricCityBlock;
pub struct MetricSqEuclidean;
pub struct MetricHamming;

/* #endregion */

/* #region Minkowski */

pub struct MetricMinkowski<T>
where
    T: ComplexFloat,
{
    pub p: T::Real,
}

impl<T> MetricMinkowski<T>
where
    T: ComplexFloat,
{
    pub fn new(p: T::Real) -> Self {
        Self { p }
    }
}

impl<T> Default for MetricMinkowski<T>
where
    T: ComplexFloat,
{
    fn default() -> Self {
        let p = T::Real::one() + T::Real::one(); // Default to p = 2.0
        Self { p }
    }
}

/* #endregion */

/* #region simple-implelentations */

#[allow(redundant_semicolons)]
#[duplicate_item(
    ImplType                       StructType           TOut      dup_reduce_op                                          dup_initialize                        dup_finalize                       ;
   [T: Float                    ] [MetricEuclidean   ] [T      ] [dist = dist + (u_i - v_i).powi(2)                   ] [                                    ] [dist = dist.sqrt();              ];
   [T: ComplexFloat<Real: Float>] [MetricMinkowski<T>] [T::Real] [dist = dist + Float::powf((u_i - v_i).abs(), self.p)] [let p_inv = T::Real::one() / self.p;] [dist = Float::powf(dist, p_inv); ];
   [T: ComplexFloat<Real: Float>] [MetricCityBlock   ] [T::Real] [dist = dist + (u_i - v_i).abs()                     ] [                                    ] [                                 ];
   [T: Float                    ] [MetricSqEuclidean ] [T      ] [dist = dist + (u_i - v_i).powi(2)                   ] [                                    ] [                                 ];
   [T: PartialEq + Copy         ] [MetricHamming     ] [f64    ] [if (u_i != v_i) { dist += 1.0 }                     ] [                                    ] [dist /= size.to_f64().unwrap();  ];
)]
impl<ImplType> MetricDistAPI<Vec<T>> for StructType {
    type Out = TOut;

    #[inline]
    fn distance<const STRIDED: bool>(
        &self,
        uv: (&Vec<T>, &Vec<T>),
        offsets: (usize, usize),
        _indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> TOut {
        let (u, v) = uv;
        let (u_offset, v_offset) = offsets;
        let mut dist = TOut::zero();
        dup_initialize;
        match STRIDED {
            false => {
                izip!(&u[u_offset..u_offset + size], &v[v_offset..v_offset + size]).for_each(
                    |(&u_i, &v_i)| {
                        dup_reduce_op;
                    },
                );
                dup_finalize;
            },
            true => {
                let (u_stride, v_stride) = strides;
                for i in 0..size {
                    let u_i = u[(u_offset as isize + i as isize * u_stride) as usize];
                    let v_i = v[(v_offset as isize + i as isize * v_stride) as usize];
                    dup_reduce_op;
                }
                dup_finalize;
            },
        }
        dist
    }
}

#[allow(redundant_semicolons)]
#[duplicate_item(
    ImplType                       StructType           TOut      dup_reduce_with_weight                                     dup_initialize                        dup_finalize                      ;
   [T: Float                    ] [MetricEuclidean   ] [T      ] [dist = dist + w * (u_i - v_i).powi(2)                   ] [                                    ] [dist = dist.sqrt();             ];
   [T: ComplexFloat<Real: Float>] [MetricMinkowski<T>] [T::Real] [dist = dist + w * Float::powf((u_i - v_i).abs(), self.p)] [let p_inv = T::Real::one() / self.p;] [dist = Float::powf(dist, p_inv);];
   [T: ComplexFloat<Real: Float>] [MetricCityBlock   ] [T::Real] [dist = dist + w * (u_i - v_i).abs()                     ] [                                    ] [                                ];
   [T: Float                    ] [MetricSqEuclidean ] [T      ] [dist = dist + w * (u_i - v_i).powi(2)                   ] [                                    ] [                                ];
   [T: PartialEq + Copy         ] [MetricHamming     ] [f64    ] [if (u_i != v_i) { dist += w }; w_sum += w;              ] [let mut w_sum = 0.0;                ] [dist /= w_sum;                  ];
)]
impl<ImplType> MetricDistWeightedAPI<Vec<T>> for StructType {
    type Weight = Vec<TOut>;
    type Out = TOut;

    #[inline]
    fn weighted_distance<const STRIDED: bool>(
        &self,
        uv: (&Vec<T>, &Vec<T>),
        weights: &Vec<TOut>,
        offsets: (usize, usize),
        _indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> TOut {
        let (u, v) = uv;
        let (u_offset, v_offset) = offsets;
        let mut dist = TOut::zero();
        dup_initialize;
        match STRIDED {
            false => {
                izip!(&u[u_offset..u_offset + size], &v[v_offset..v_offset + size], weights)
                    .for_each(|(&u_i, &v_i, &w)| {
                        dup_reduce_with_weight;
                    });
                dup_finalize;
            },
            true => {
                let (u_stride, v_stride) = strides;
                for i in 0..size {
                    let u_i = u[(u_offset as isize + i as isize * u_stride) as usize];
                    let v_i = v[(v_offset as isize + i as isize * v_stride) as usize];
                    let w = weights[i];
                    dup_reduce_with_weight;
                }
                dup_finalize;
            },
        }
        dist
    }
}

/* #endregion */
