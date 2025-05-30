use num::Float;
use rstsr_core::prelude_dev::*;

/// Distance metric API trait.
///
/// # Generic Arguments
/// * `T` - The type of the distance metric result (e.g., f32, f64).
/// * `V` - The type of the vectors (`Vec<T>` in general, but can be other)
pub trait MetricDistAPI<T, V> {
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
    fn distance<const WEIGHTED: bool, const STRIDED: bool>(
        &self,
        uv: (&V, &V),
        weights: Option<&Self::Weight>,
        offsets: (usize, usize),
        indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> Self::Out;
}

/* #region Euclidean */

pub struct MetricEuclidean;

impl<T> MetricDistAPI<T, Vec<T>> for MetricEuclidean
where
    T: Float,
{
    type Weight = Vec<T>;
    type Out = T;

    #[inline]
    fn distance<const WEIGHTED: bool, const STRIDED: bool>(
        &self,
        uv: (&Vec<T>, &Vec<T>),
        weights: Option<&Vec<T>>,
        offsets: (usize, usize),
        _indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
    ) -> T {
        let (u, v) = uv;
        let (u_offset, v_offset) = offsets;
        let mut dist = T::zero();
        match (WEIGHTED, STRIDED) {
            (false, false) => {
                izip!(&u[u_offset..u_offset + size], &v[v_offset..v_offset + size]).for_each(
                    |(&u_i, &v_i)| {
                        dist = dist + (u_i - v_i).powi(2);
                    },
                );
                dist = dist.sqrt();
            },
            (true, false) => {
                izip!(
                    &u[u_offset..u_offset + size],
                    &v[v_offset..v_offset + size],
                    weights.unwrap()
                )
                .for_each(|(&u_i, &v_i, &w)| {
                    dist = dist + w * (u_i - v_i).powi(2);
                });
                dist = dist.sqrt();
            },
            (false, true) => {
                let (u_stride, v_stride) = strides;
                for i in 0..size {
                    let u_i = u[(u_offset as isize + i as isize * u_stride) as usize];
                    let v_i = v[(v_offset as isize + i as isize * v_stride) as usize];
                    dist = dist + (u_i - v_i).powi(2);
                }
                dist = dist.sqrt();
            },
            (true, true) => {
                let (u_stride, v_stride) = strides;
                for i in 0..size {
                    let u_i = u[(u_offset as isize + i as isize * u_stride) as usize];
                    let v_i = v[(v_offset as isize + i as isize * v_stride) as usize];
                    let w = weights.unwrap()[i];
                    dist = dist + w * (u_i - v_i).powi(2);
                }
                dist = dist.sqrt();
            },
        }
        dist
    }
}
