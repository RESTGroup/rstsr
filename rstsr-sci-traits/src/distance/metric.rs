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

    /// Initializes the distance metric.
    ///
    /// In most cases, this is a no-op, but some metrics may require (cosine,
    /// for example).
    #[inline]
    fn initialize(&mut self, _xa: &V, _la: &Layout<Ix2>, _xb: &V, _lb: &Layout<Ix2>) {}
}

/// Distance metric API trait.
///
/// # Generic Arguments
/// * `T` - The type of the distance metric result (e.g., f32, f64).
/// * `V` - The type of the vectors (`Vec<T>` in general, but can be other)
pub trait MetricDistWeightedAPI<V>: MetricDistAPI<V> {
    // The type of the weights, should be vector type of `Self::Out`.
    type Weight;

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
        offsets: (usize, usize),
        indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
        weights: &Self::Weight,
        weights_sum: Self::Out,
    ) -> Self::Out;
}

/* #region zero-size distance types */

// float-type distance metrics

pub struct MetricEuclidean;
pub struct MetricCityBlock;
pub struct MetricSqEuclidean;
pub struct MetricCanberra;
pub struct MetricBrayCurtis;
pub struct MetricChebyshev;

// boolean-like distance metrics

pub struct MetricHamming;
pub struct MetricJaccard;
pub struct MetricYule;
pub struct MetricDice;
pub struct MetricRogersTanimoto;
pub struct MetricRussellRao;
pub struct MetricSokalSneath;

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

#[inline]
fn inner_jaccard(u: bool, v: bool, num: &mut f64, denom: &mut f64) {
    if u ^ v {
        *num += 1.0;
    }
    if u || v {
        *denom += 1.0;
    }
}

#[inline]
fn inner_jaccard_w(u: bool, v: bool, w: f64, num: &mut f64, denom: &mut f64) {
    if u ^ v {
        *num += w;
    }
    if u || v {
        *denom += w;
    }
}

#[inline]
fn inner_canberra<T: ComplexFloat<Real: Float>>(u: T, v: T, dist: &mut T::Real) {
    let snum = (u - v).abs();
    let sdenom = u.abs() + v.abs();
    if sdenom != T::Real::zero() {
        *dist = *dist + snum / sdenom;
    }
}

#[inline]
fn inner_canberra_w<T: ComplexFloat<Real: Float>>(u: T, v: T, w: T::Real, dist: &mut T::Real) {
    let snum = (u - v).abs();
    let sdenom = u.abs() + v.abs();
    if sdenom != T::Real::zero() {
        *dist = *dist + w * snum / sdenom;
    }
}

#[inline]
fn inner_bray_curtis<T: ComplexFloat<Real: Float>>(u: T, v: T, s: &mut [T::Real; 2]) {
    let [s1, s2] = s;
    *s1 = *s1 + (u - v).abs();
    *s2 = *s2 + (u + v).abs();
}

#[inline]
fn inner_bray_curtis_w<T: ComplexFloat<Real: Float>>(u: T, v: T, w: T::Real, s: &mut [T::Real; 2]) {
    let [s1, s2] = s;
    *s1 = *s1 + w * (u - v).abs();
    *s2 = *s2 + w * (u + v).abs();
}

#[inline]
fn inner_chebyshev<T: ComplexFloat<Real: Float>>(u: T, v: T, dist: &mut T::Real) {
    *dist = (*dist).max((u - v).abs());
}

#[inline]
fn inner_chebyshev_w<T: ComplexFloat<Real: Float>>(u: T, v: T, w: T::Real, dist: &mut T::Real) {
    if w != T::Real::zero() {
        *dist = (*dist).max((u - v).abs());
    }
}

#[inline]
fn inner_yule(u: bool, v: bool, n: &mut [f64; 4]) {
    let [ntt, nff, nft, ntf] = n;
    match (u, v) {
        (true, true) => *ntt += 1.0,
        (false, false) => *nff += 1.0,
        (true, false) => *nft += 1.0,
        (false, true) => *ntf += 1.0,
    }
}

#[inline]
fn inner_yule_w(u: bool, v: bool, w: f64, n: &mut [f64; 4]) {
    let [ntt, nff, nft, ntf] = n;
    match (u, v) {
        (true, true) => *ntt += w,
        (false, false) => *nff += w,
        (true, false) => *nft += w,
        (false, true) => *ntf += w,
    }
}

#[inline]
fn inner_dice(u: bool, v: bool, n: &mut [f64; 2]) {
    let [ntt, ndiff] = n;
    match (u, v) {
        (true, true) => *ntt += 1.0,
        (false, false) => (),
        (true, false) => *ndiff += 1.0,
        (false, true) => *ndiff += 1.0,
    }
}

#[inline]
fn inner_dice_w(u: bool, v: bool, w: f64, n: &mut [f64; 2]) {
    let [ntt, ndiff] = n;
    match (u, v) {
        (true, true) => *ntt += w,
        (false, false) => (),
        (true, false) => *ndiff += w,
        (false, true) => *ndiff += w,
    }
}

#[allow(redundant_semicolons)]
#[allow(unused_assignments)]
#[duplicate_item(
    T      ImplType                       StructType             TOut      dup_reduce_op                                          dup_initialize                        dup_finalize                                                   ;
   [T   ] [T: Float                    ] [MetricEuclidean     ] [T      ] [dist = dist + (u_i - v_i).powi(2)                   ] [                                   ] [dist = dist.sqrt()                                            ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricMinkowski<T>  ] [T::Real] [dist = dist + Float::powf((u_i - v_i).abs(), self.p)] [let p_inv = T::Real::one() / self.p] [dist = Float::powf(dist, p_inv)                               ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricCityBlock     ] [T::Real] [dist = dist + (u_i - v_i).abs()                     ] [                                   ] [                                                              ];
   [T   ] [T: Float                    ] [MetricSqEuclidean   ] [T      ] [dist = dist + (u_i - v_i).powi(2)                   ] [                                   ] [                                                              ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricCanberra      ] [T::Real] [inner_canberra(u_i, v_i, &mut dist)                 ] [                                   ] [                                                              ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricChebyshev     ] [T::Real] [inner_chebyshev(u_i, v_i, &mut dist)                ] [                                   ] [                                                              ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricBrayCurtis    ] [T::Real] [inner_bray_curtis(u_i, v_i, &mut s)                 ] [let mut s = [T::Real::zero(); 2]   ] [dist = s[0] / s[1]                                            ];
   [T   ] [T: PartialEq + Copy         ] [MetricHamming       ] [f64    ] [if (u_i != v_i) { dist += 1.0 }                     ] [                                   ] [dist /= size.to_f64().unwrap()                                ];
   [bool] [                            ] [MetricJaccard       ] [f64    ] [inner_jaccard(u_i, v_i, &mut dist, &mut denom)      ] [let mut denom = 0.0                ] [dist /= denom                                                 ];
   [bool] [                            ] [MetricYule          ] [f64    ] [inner_yule(u_i, v_i, &mut n)                        ] [let mut n = [0.0; 4]               ] [dist = (2. * n[2] * n[3]) / (n[0] * n[1] + n[2] * n[3])       ];
   [bool] [                            ] [MetricDice          ] [f64    ] [inner_dice(u_i, v_i, &mut n)                        ] [let mut n = [0.0; 2]               ] [dist = n[1] / (2. * n[0] + n[1])                              ];
   [bool] [                            ] [MetricRogersTanimoto] [f64    ] [inner_dice(u_i, v_i, &mut n)                        ] [let mut n = [0.0; 2]               ] [dist = (2. * n[1]) / (size.to_f64().unwrap() + n[1])          ];
   [bool] [                            ] [MetricRussellRao    ] [f64    ] [if (u_i && v_i) { ntt += 1.; }                      ] [let mut ntt = 0.0                  ] [dist = (size.to_f64().unwrap() - ntt) / size.to_f64().unwrap()];
   [bool] [                            ] [MetricSokalSneath   ] [f64    ] [inner_dice(u_i, v_i, &mut n)                        ] [let mut n = [0.0; 2]               ] [dist = (2. * n[1]) / (2. * n[1] + n[0])                       ];
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
#[allow(unused_variables)]
#[allow(unused_assignments)]
#[duplicate_item(
    T      ImplType                       StructType             TOut      dup_reduce_with_weight                                     dup_initialize                        dup_finalize                                            ;
   [T   ] [T: Float                    ] [MetricEuclidean     ] [T      ] [dist = dist + w * (u_i - v_i).powi(2)                   ] [                                   ] [dist = dist.sqrt()                                     ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricMinkowski<T>  ] [T::Real] [dist = dist + w * Float::powf((u_i - v_i).abs(), self.p)] [let p_inv = T::Real::one() / self.p] [dist = Float::powf(dist, p_inv)                        ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricCityBlock     ] [T::Real] [dist = dist + w * (u_i - v_i).abs()                     ] [                                   ] [                                                       ];
   [T   ] [T: Float                    ] [MetricSqEuclidean   ] [T      ] [dist = dist + w * (u_i - v_i).powi(2)                   ] [                                   ] [                                                       ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricCanberra      ] [T::Real] [inner_canberra_w(u_i, v_i, w, &mut dist)                ] [                                   ] [                                                       ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricChebyshev     ] [T::Real] [inner_chebyshev_w(u_i, v_i, w, &mut dist)               ] [                                   ] [                                                       ];
   [T   ] [T: ComplexFloat<Real: Float>] [MetricBrayCurtis    ] [T::Real] [inner_bray_curtis_w(u_i, v_i, w, &mut s)                ] [let mut s = [T::Real::zero(); 2]   ] [dist = s[0] / s[1]                                     ];
   [T   ] [T: PartialEq + Copy         ] [MetricHamming       ] [f64    ] [if (u_i != v_i) { dist += w }                           ] [                                   ] [dist /= weights_sum                                    ];
   [bool] [                            ] [MetricJaccard       ] [f64    ] [inner_jaccard_w(u_i, v_i, w, &mut dist, &mut denom)     ] [let mut denom = 0.0                ] [dist /= denom                                          ];
   [bool] [                            ] [MetricYule          ] [f64    ] [inner_yule_w(u_i, v_i, w, &mut n)                       ] [let mut n = [0.0; 4]               ] [dist = (2. * n[2] * n[3]) / (n[0] * n[1] + n[2] * n[3])];
   [bool] [                            ] [MetricDice          ] [f64    ] [inner_dice_w(u_i, v_i, w, &mut n)                       ] [let mut n = [0.0; 2]               ] [dist = n[1] / (2. * n[0] + n[1])                       ];
   [bool] [                            ] [MetricRogersTanimoto] [f64    ] [inner_dice_w(u_i, v_i, w, &mut n)                       ] [let mut n = [0.0; 2]               ] [dist = (2. * n[1]) / (weights_sum + n[1])              ];
   [bool] [                            ] [MetricRussellRao    ] [f64    ] [if (u_i && v_i) { ntt += w; }                           ] [let mut ntt = 0.0                  ] [dist = (weights_sum - ntt) / weights_sum               ];
   [bool] [                            ] [MetricSokalSneath   ] [f64    ] [inner_dice_w(u_i, v_i, w, &mut n)                       ] [let mut n = [0.0; 2]               ] [dist = (2. * n[1]) / (2. * n[1] + n[0])                ];
)]
impl<ImplType> MetricDistWeightedAPI<Vec<T>> for StructType {
    type Weight = Vec<TOut>;

    #[inline]
    fn weighted_distance<const STRIDED: bool>(
        &self,
        uv: (&Vec<T>, &Vec<T>),
        offsets: (usize, usize),
        _indices: (usize, usize),
        strides: (isize, isize),
        size: usize,
        weights: &Vec<TOut>,
        weights_sum: TOut,
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
