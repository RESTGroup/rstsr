use core::str::FromStr;
use derive_builder::Builder;
use rstsr_blas_traits::prelude::BlasFloat;
use rstsr_core::prelude_dev::*;

/* #region trait and fn definitions */

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EigAPI            ] [eig             ] [eig_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [EigvalsAPI        ] [eigvals         ] [eigvals_f         ];
   [EigvalshAPI       ] [eigvalsh        ] [eigvalsh_f        ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [LstsqAPI          ] [lstsq           ] [lstsq_f           ];
   [MatrixNormAPI     ] [matrix_norm     ] [matrix_norm_f     ];
   [MatrixPowerAPI    ] [matrix_power    ] [matrix_power_f    ];
   [MatrixRankAPI     ] [matrix_rank     ] [matrix_rank_f     ];
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QrAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
   [VectorNormAPI     ] [vector_norm     ] [vector_norm_f     ];
)]
pub trait LinalgAPI<Inp> {
    type Out;
    fn func_f(self) -> Result<Self::Out>;
    fn func(self) -> Self::Out
    where
        Self: Sized,
    {
        Self::func_f(self).rstsr_unwrap()
    }
}

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EigAPI            ] [eig             ] [eig_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [EigvalsAPI        ] [eigvals         ] [eigvals_f         ];
   [EigvalshAPI       ] [eigvalsh        ] [eigvalsh_f        ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [LstsqAPI          ] [lstsq           ] [lstsq_f           ];
   [MatrixNormAPI     ] [matrix_norm     ] [matrix_norm_f     ];
   [MatrixPowerAPI    ] [matrix_power    ] [matrix_power_f    ];
   [MatrixRankAPI     ] [matrix_rank     ] [matrix_rank_f     ];
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QrAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
   [VectorNormAPI     ] [vector_norm     ] [vector_norm_f     ];
)]
pub fn func_f<Args, Inp>(args: Args) -> Result<<Args as LinalgAPI<Inp>>::Out>
where
    Args: LinalgAPI<Inp>,
{
    Args::func_f(args)
}

#[duplicate_item(
    LinalgAPI            func               func_f             ;
   [CholeskyAPI       ] [cholesky        ] [cholesky_f        ];
   [DetAPI            ] [det             ] [det_f             ];
   [EigAPI            ] [eig             ] [eig_f             ];
   [EighAPI           ] [eigh            ] [eigh_f            ];
   [EigvalsAPI        ] [eigvals         ] [eigvals_f         ];
   [EigvalshAPI       ] [eigvalsh        ] [eigvalsh_f        ];
   [InvAPI            ] [inv             ] [inv_f             ];
   [LstsqAPI          ] [lstsq           ] [lstsq_f           ];
   [MatrixNormAPI     ] [matrix_norm     ] [matrix_norm_f     ];
   [MatrixPowerAPI    ] [matrix_power    ] [matrix_power_f    ];
   [MatrixRankAPI     ] [matrix_rank     ] [matrix_rank_f     ];
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QrAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
   [VectorNormAPI     ] [vector_norm     ] [vector_norm_f     ];
)]
pub fn func<Args, Inp>(args: Args) -> <Args as LinalgAPI<Inp>>::Out
where
    Args: LinalgAPI<Inp>,
{
    Args::func(args)
}

/* #endregion */

/* #region eig */

/// Result of a general (non-Hermitian) eigenproblem: the eigenvalues `W` and
/// the left (`VL`) and right (`VR`) eigenvector blocks of the requested sides.
///
/// One type describes every form of [`eig`]: `W` is the spectrum (the
/// homogeneous form carries `(alpha, beta)` instead), and `VL`/`VR` are
/// whatever the form returns. `left` is `None` on the faer device (right
/// eigenvectors only), and every side that was not requested is `None`. The fields are
/// public, and the conversions below destructure the common shapes:
///
/// - `(W, VL, VR)` — always available, type-preserving.
/// - `(W, T)` — from `EigResult<W, Option<T>, Option<T>>` when exactly one side is present (the
///   default form gives `(w, vr)`); panics otherwise.
/// - `(W, T, T)` — from `EigResult<W, Option<T>, Option<T>>` when both sides are present; panics
///   otherwise.
///
/// See also [`eig`], [`eigvals`], [`EigSide`].
pub struct EigResult<W, VL, VR> {
    pub eigenvalues: W,
    pub left: VL,
    pub right: VR,
}

impl<W, VL, VR> From<(W, VL, VR)> for EigResult<W, VL, VR> {
    fn from((eigenvalues, left, right): (W, VL, VR)) -> Self {
        Self { eigenvalues, left, right }
    }
}

impl<W, VL, VR> From<EigResult<W, VL, VR>> for (W, VL, VR) {
    fn from(eig_result: EigResult<W, VL, VR>) -> Self {
        (eig_result.eigenvalues, eig_result.left, eig_result.right)
    }
}

impl<W, T> From<EigResult<W, Option<T>, Option<T>>> for (W, T) {
    /// Destructures a one-sided result into `(eigenvalues, eigenvectors)`.
    ///
    /// # Panics
    ///
    /// Panics when both sides are present (use the `(W, T, T)` conversion) or
    /// neither is (the spectrum is still available as `.eigenvalues`).
    fn from(eig_result: EigResult<W, Option<T>, Option<T>>) -> Self {
        let EigResult { eigenvalues, left, right } = eig_result;
        match (left, right) {
            (Some(v), None) => (eigenvalues, v),
            (None, Some(v)) => (eigenvalues, v),
            (Some(_), Some(_)) => panic!("EigResult: both eigenvector sides are present; convert to (W, T, T) instead"),
            (None, None) => panic!("EigResult: no eigenvector side is present; read `eigenvalues` directly"),
        }
    }
}

impl<W, T> From<EigResult<W, Option<T>, Option<T>>> for (W, T, T) {
    /// Destructures a two-sided result into `(eigenvalues, left, right)`.
    ///
    /// # Panics
    ///
    /// Panics unless both sides are present.
    fn from(eig_result: EigResult<W, Option<T>, Option<T>>) -> Self {
        let EigResult { eigenvalues, left, right } = eig_result;
        match (left, right) {
            (Some(vl), Some(vr)) => (eigenvalues, vl, vr),
            (Some(_), None) => panic!("EigResult: only the left eigenvectors are present; convert to (W, T) instead"),
            (None, Some(_)) => panic!("EigResult: only the right eigenvectors are present; convert to (W, T) instead"),
            (None, None) => panic!("EigResult: no eigenvector side is present; read `eigenvalues` directly"),
        }
    }
}

/// Which eigenvector blocks [`eig`] returns.
///
/// The default is [`EigSide::Right`] (the NumPy `eig` convention
/// `A v = λ v`). [`EigSide::Left`] returns `u` with
/// `uᴴ A = λ uᴴ` (the SciPy `left=True, right=False` form),
/// [`EigSide::Both`] returns the two blocks, and [`EigSide::None`] returns the
/// spectrum alone. On the BLAS devices [`eig`] accepts the enum, the
/// `"none"`/`"left"`/`"right"`/`"both"` strings, and [`FlagSide::L`]/
/// [`FlagSide::R`] in the side slot; the faer device takes the bare `eig(a)`
/// (right eigenvectors) only.
///
/// `#[non_exhaustive]`: further sides (if any) can be added without a
/// breaking change.
///
/// See also [`eig`], [`eigvals`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EigSide {
    /// Neither block: the eigenvalues only.
    None,
    /// Left eigenvectors only: `u(j)ᴴ A = λ(j) u(j)ᴴ`.
    Left,
    /// Right eigenvectors only: `A v(j) = λ(j) v(j)` (the default).
    #[default]
    Right,
    /// Both the left and the right eigenvectors.
    Both,
}

impl EigSide {
    /// Whether the left eigenvector block is requested.
    pub fn wants_left(self) -> bool {
        matches!(self, EigSide::Left | EigSide::Both)
    }

    /// Whether the right eigenvector block is requested.
    pub fn wants_right(self) -> bool {
        matches!(self, EigSide::Right | EigSide::Both)
    }
}

impl TryFrom<&str> for EigSide {
    type Error = Error;
    fn try_from(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(EigSide::None),
            "left" => Ok(EigSide::Left),
            "right" => Ok(EigSide::Right),
            "both" => Ok(EigSide::Both),
            _ => Err(rstsr_error!(InvalidValue, "invalid eig side '{s}'")),
        }
    }
}

impl FromStr for EigSide {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s)
    }
}

impl From<FlagSide> for EigSide {
    fn from(side: FlagSide) -> Self {
        match side {
            FlagSide::L => EigSide::Left,
            FlagSide::R => EigSide::Right,
        }
    }
}

/// Marker for the SciPy `homogeneous_eigvals=True` form of [`eig`] and
/// [`eigvals`] (BLAS devices only): the eigenvalues come as the `(alpha,
/// beta)` pair stacked on a new second-to-last axis (shape `(..., 2, M)`,
/// `(2, M, ...)` under a column-major device), so that
/// `lambda = alpha / beta` is never divided out.
///
/// For the standard problem `beta` is identically `1`; for the generalized
/// problem `A v = lambda B v` the pair is the LAPACK `(alpha, beta)`
/// representation (a singular `B` gives `beta(j) = 0` for some `j`, without an `inf` appearing).
///
/// See also [`eig`], [`eigvals`], [`EigResult`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Homogeneous;

/* #endregion */

/* #region eigh */

pub struct EighResult<W, V> {
    pub eigenvalues: W,
    pub eigenvectors: V,
}

impl<W, V> From<(W, V)> for EighResult<W, V> {
    fn from((vals, vecs): (W, V)) -> Self {
        Self { eigenvalues: vals, eigenvectors: vecs }
    }
}

impl<W, V> From<EighResult<W, V>> for (W, V) {
    fn from(eigh_result: EighResult<W, V>) -> Self {
        (eigh_result.eigenvalues, eigh_result.eigenvectors)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct EighArgs_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into, strip_option), default = "None")]
    pub b: Option<TensorReference<'b, T, B, Ix2>>,

    #[builder(setter(into), default = "None")]
    pub uplo: Option<FlagUpLo>,
    #[builder(setter(into), default = false)]
    pub eigvals_only: bool,
    #[builder(setter(into), default = 1)]
    pub eig_type: i32,
    #[builder(setter(into, strip_option), default = "None")]
    pub subset_by_index: Option<(usize, usize)>,
    #[builder(setter(into, strip_option), default = "None")]
    pub subset_by_value: Option<(T::Real, T::Real)>,
    #[builder(setter(into, strip_option), default = "None")]
    pub driver: Option<&'static str>,
}

pub type EighArgs<'a, 'b, B, T> = EighArgs_Builder<'a, 'b, B, T>;

/* #endregion */

/* #region pinv */

pub struct PinvResult<T> {
    pub pinv: T,
    pub rank: usize,
}

impl<T> From<(T, usize)> for PinvResult<T> {
    fn from((pinv, rank): (T, usize)) -> Self {
        Self { pinv, rank }
    }
}

impl<T> From<PinvResult<T>> for (T, usize) {
    fn from(pinv_result: PinvResult<T>) -> Self {
        (pinv_result.pinv, pinv_result.rank)
    }
}

/// Relative tolerance for [`pinv`](crate::traits_def::pinv): singular values at
/// or below `atol + rtol * s_max` are set to zero.
///
/// A scalar carries no device; an array does — so the two cannot share one
/// representation, and the array's device is checked against the input's before
/// the pseudoinverse is computed. `None` selects the default `max(M, N) * eps`.
///
/// **Order notice.** The input's two matrix axes are its *leading* axes under a
/// row-major device but its *trailing* axes under a column-major one; the batch
/// is whatever remains, and an [`Array`](PinvRtol::Array) tolerance is broadcast
/// against the batch in that same device order. On a row-major device this is
/// NumPy's trailing alignment against `shape(x)[:-2]`; on a column-major device
/// the matrix axes are the right-most and the tolerance aligns from the opposite
/// end.
///
/// `TR` is the input's real scalar type (`T::Real`) and `B` its device. `From`
/// converts the scalar, array, `()`, and `None` forms, so `pinv((x, 1e-8))` (an
/// `f64` input), `pinv((x, rtol_array))`, and `pinv((x, None))` all resolve. A
/// scalar is taken as the input's own real type — with an `f32` input write
/// `1e-8_f32` (a bare `1e-8` is `f64` and does not convert).
pub enum PinvRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    /// The default tolerance `max(M, N) * eps`.
    None,
    /// One tolerance for every matrix — a scalar of the input's real type.
    Float(TR),
    /// One tolerance per matrix, broadcast against the batch in device order.
    Array(Tensor<TR, B, IxD>),
}

// not derived: `#[derive(Default)]` would add spurious `TR: Default, B: Default` bounds
#[allow(clippy::derivable_impls)]
impl<TR, B> Default for PinvRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn default() -> Self {
        Self::None
    }
}

impl<TR, B> From<Tensor<TR, B, IxD>> for PinvRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn from(rtol: Tensor<TR, B, IxD>) -> Self {
        Self::Array(rtol)
    }
}

// the scalar overload is written per real type, not as a generic `From<TR>`:
// a generic impl would overlap `From<()>`/`From<Option<()>>` at `TR = ()`
impl<B> From<f32> for PinvRtol<f32, B>
where
    B: DeviceRawAPI<f32>,
{
    fn from(rtol: f32) -> Self {
        Self::Float(rtol)
    }
}

impl<B> From<f64> for PinvRtol<f64, B>
where
    B: DeviceRawAPI<f64>,
{
    fn from(rtol: f64) -> Self {
        Self::Float(rtol)
    }
}

impl<TR, B> From<()> for PinvRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn from(_: ()) -> Self {
        Self::None
    }
}

impl<TR, B> From<Option<()>> for PinvRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    /// `None` is equivalent to `()`: both request the default tolerance.
    fn from(_: Option<()>) -> Self {
        Self::None
    }
}

/* #endregion */

/* #region lstsq */

/// Result of a least-squares solve: the solution `x`, the residual
/// sum-of-squares per right-hand side, the numerical rank and the singular
/// values.
///
/// The shapes follow NumPy and SciPy. With a matrix right-hand side `b` the
/// solution is `(..., N, K)` and `residuals` is `(..., K)`; a vector `b`
/// squeezes the last axis, so `residuals` takes the batch shape (0-d when
/// unbatched). Residuals are real even for complex inputs, are `NaN` on
/// slices whose rank is below `N`, and are empty when `M <= N`. `rank` keeps
/// one value per matrix slice. The conversions below destructure the four
/// fields.
///
/// See also [`lstsq`], [`LstsqAPI`].
pub struct LstsqResult<X, Res, Rank, S> {
    pub x: X,
    pub residuals: Res,
    pub rank: Rank,
    pub s: S,
}

impl<X, Res, Rank, S> From<(X, Res, Rank, S)> for LstsqResult<X, Res, Rank, S> {
    fn from((x, residuals, rank, s): (X, Res, Rank, S)) -> Self {
        Self { x, residuals, rank, s }
    }
}

impl<X, Res, Rank, S> From<LstsqResult<X, Res, Rank, S>> for (X, Res, Rank, S) {
    fn from(lstsq_result: LstsqResult<X, Res, Rank, S>) -> Self {
        (lstsq_result.x, lstsq_result.residuals, lstsq_result.rank, lstsq_result.s)
    }
}

/* #endregion */

/* #region slogdet */

pub struct SLogDetResult<S, L> {
    pub sign: S,
    pub logabsdet: L,
}

impl<S, L> From<(S, L)> for SLogDetResult<S, L> {
    fn from((sign, logabsdet): (S, L)) -> Self {
        Self { sign, logabsdet }
    }
}

impl<S, L> From<SLogDetResult<S, L>> for (S, L) {
    fn from(slogdet_result: SLogDetResult<S, L>) -> Self {
        (slogdet_result.sign, slogdet_result.logabsdet)
    }
}

/* #endregion */

/* #region svd */

pub struct SVDResult<U, S, Vt> {
    pub u: U,
    pub s: S,
    pub vt: Vt,
}

impl<U, S, Vt> From<(U, S, Vt)> for SVDResult<U, S, Vt> {
    fn from((u, s, vt): (U, S, Vt)) -> Self {
        Self { u, s, vt }
    }
}

impl<U, S, Vt> From<SVDResult<U, S, Vt>> for (U, S, Vt) {
    fn from(svd_result: SVDResult<U, S, Vt>) -> Self {
        (svd_result.u, svd_result.s, svd_result.vt)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct SVDArgs_<'a, B, T>
where
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into), default = "Some(true)")]
    pub full_matrices: Option<bool>,
    #[builder(setter(into), default = "None")]
    pub driver: Option<&'static str>,
}

pub type SVDArgs<'a, B, T> = SVDArgs_Builder<'a, B, T>;

/* #endregion */

/* #region qr */

/// QR factorization result: the orthogonal/unitary `Q` and upper-triangular `R`.
pub struct QrResult<Q, R> {
    pub q: Q,
    pub r: R,
}

impl<Q, R> From<(Q, R)> for QrResult<Q, R> {
    fn from((q, r): (Q, R)) -> Self {
        Self { q, r }
    }
}

impl<Q, R> From<QrResult<Q, R>> for (Q, R) {
    fn from(qr_result: QrResult<Q, R>) -> Self {
        (qr_result.q, qr_result.r)
    }
}

/// The [`qr`] factorization mode, named by the array-API string `'reduced'`
/// (default) or `'complete'`. Both the faer device and the BLAS devices
/// implement these two modes.
///
/// `#[non_exhaustive]`: only these two modes exist today, but further modes can
/// be added without a breaking change.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum QrMode {
    /// Reduced (thin): `Q` is `(..., M, K)`, `R` is `(..., K, N)` for `K = min(M, N)`.
    #[default]
    Reduced,
    /// Complete: `Q` is `(..., M, M)`, `R` is `(..., M, N)`.
    Complete,
}

impl TryFrom<&str> for QrMode {
    type Error = Error;
    fn try_from(s: &str) -> Result<Self> {
        match s {
            "reduced" => Ok(QrMode::Reduced),
            "complete" => Ok(QrMode::Complete),
            _ => Err(rstsr_error!(InvalidValue, "invalid qr mode '{s}'")),
        }
    }
}

impl FromStr for QrMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s)
    }
}

/* #endregion */

/* #region norm */

/// Norm order shared by [`matrix_norm`] and [`vector_norm`].
///
/// `None` is the default and resolves to each operation's own default order
/// (`'fro'` for [`matrix_norm`], `2` for [`vector_norm`]), so the same value
/// names different orders in the two operations. `P` holds a numeric order;
/// `vector_norm` accepts any real `p`, `matrix_norm` only `±1`/`±2`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum NormOrd {
    /// The operation's own default order (`'fro'` for matrices, `2` for vectors).
    #[default]
    None,
    /// Frobenius norm (matrix only).
    Fro,
    /// Nuclear norm (matrix only).
    Nuc,
    /// A numeric order `p`.
    P(f64),
    /// `+inf`.
    Inf,
    /// `-inf`.
    NegInf,
}

impl NormOrd {
    /// Resolve to a [`matrix_norm`] order: `None` becomes [`NormOrd::Fro`], and a
    /// numeric order outside `±1`/`±2` is rejected.
    pub fn resolve_matrix(self) -> Result<Self> {
        match self {
            NormOrd::None => Ok(NormOrd::Fro),
            NormOrd::Fro | NormOrd::Nuc | NormOrd::Inf | NormOrd::NegInf => Ok(self),
            NormOrd::P(p) if p == 1.0 || p == -1.0 || p == 2.0 || p == -2.0 => Ok(NormOrd::P(p)),
            NormOrd::P(p) => Err(rstsr_error!(InvalidValue, "matrix_norm: invalid norm order {p} for a matrix")),
        }
    }

    /// Resolve to a [`vector_norm`] order: `None` becomes the `2`-norm, and the
    /// matrix-only orders (`'fro'`/`'nuc'`) are rejected.
    pub fn resolve_vector(self) -> Result<Self> {
        match self {
            NormOrd::None => Ok(NormOrd::P(2.0)),
            NormOrd::P(_) | NormOrd::Inf | NormOrd::NegInf => Ok(self),
            NormOrd::Fro => Err(rstsr_error!(InvalidValue, "vector_norm: 'fro' is not a vector norm order")),
            NormOrd::Nuc => Err(rstsr_error!(InvalidValue, "vector_norm: 'nuc' is not a vector norm order")),
        }
    }
}

impl TryFrom<()> for NormOrd {
    type Error = Error;
    fn try_from(_: ()) -> Result<Self> {
        Ok(NormOrd::None)
    }
}

impl TryFrom<Option<()>> for NormOrd {
    type Error = Error;
    fn try_from(_: Option<()>) -> Result<Self> {
        Ok(NormOrd::None)
    }
}

impl TryFrom<&str> for NormOrd {
    type Error = Error;
    fn try_from(s: &str) -> Result<Self> {
        match s {
            "fro" => Ok(NormOrd::Fro),
            "nuc" => Ok(NormOrd::Nuc),
            _ => Err(rstsr_error!(InvalidValue, "invalid norm order '{s}'")),
        }
    }
}

impl FromStr for NormOrd {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s)
    }
}

impl TryFrom<isize> for NormOrd {
    type Error = Error;
    fn try_from(v: isize) -> Result<Self> {
        Ok(NormOrd::P(v as f64))
    }
}

impl TryFrom<f64> for NormOrd {
    type Error = Error;
    fn try_from(v: f64) -> Result<Self> {
        Ok(if v == f64::INFINITY {
            NormOrd::Inf
        } else if v == f64::NEG_INFINITY {
            NormOrd::NegInf
        } else {
            NormOrd::P(v)
        })
    }
}

/// Arguments for [`matrix_norm`]: the norm order and `keepdims`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixNormArgs {
    pub ord: NormOrd,
    pub keepdims: bool,
}

/// Arguments for [`vector_norm`]: the norm order, the axis or axes to reduce
/// (`AxesIndex::None` reduces all of them), and `keepdims`.
#[derive(Debug, Clone)]
pub struct VectorNormArgs {
    pub ord: NormOrd,
    pub axes: AxesIndex<isize>,
    pub keepdims: bool,
}

impl Default for VectorNormArgs {
    fn default() -> Self {
        Self { ord: NormOrd::None, axes: AxesIndex::None, keepdims: false }
    }
}

macro_rules! impl_norm_args_default {
    ($Args: ident) => {
        impl TryFrom<()> for $Args {
            type Error = Error;
            fn try_from(_: ()) -> Result<Self> {
                Ok(Self::default())
            }
        }

        impl TryFrom<Option<()>> for $Args {
            type Error = Error;
            fn try_from(_: Option<()>) -> Result<Self> {
                Ok(Self::default())
            }
        }

        impl TryFrom<bool> for $Args {
            type Error = Error;
            fn try_from(keepdims: bool) -> Result<Self> {
                Ok(Self { keepdims, ..Default::default() })
            }
        }
    };
}

impl_norm_args_default!(MatrixNormArgs);
impl_norm_args_default!(VectorNormArgs);

impl TryFrom<NormOrd> for MatrixNormArgs {
    type Error = Error;
    fn try_from(ord: NormOrd) -> Result<Self> {
        Ok(Self { ord, keepdims: false })
    }
}

impl TryFrom<(NormOrd, bool)> for MatrixNormArgs {
    type Error = Error;
    fn try_from((ord, keepdims): (NormOrd, bool)) -> Result<Self> {
        Ok(Self { ord, keepdims })
    }
}

impl TryFrom<NormOrd> for VectorNormArgs {
    type Error = Error;
    fn try_from(ord: NormOrd) -> Result<Self> {
        Ok(Self { ord, ..Default::default() })
    }
}

impl TryFrom<(NormOrd, bool)> for VectorNormArgs {
    type Error = Error;
    fn try_from((ord, keepdims): (NormOrd, bool)) -> Result<Self> {
        Ok(Self { ord, keepdims, ..Default::default() })
    }
}

/// `ord` value forms accepted anywhere an order is expected: the enum itself,
/// `"fro"`/`"nuc"`, an integer, or a float (`±inf` included).
macro_rules! impl_matrix_norm_args_ord {
    ($($t: ty),* $(,)?) => {$(
        impl TryFrom<$t> for MatrixNormArgs {
            type Error = Error;
            fn try_from(ord: $t) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, keepdims: false })
            }
        }

        impl TryFrom<($t, bool)> for MatrixNormArgs {
            type Error = Error;
            fn try_from((ord, keepdims): ($t, bool)) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, keepdims })
            }
        }
    )*};
}

impl_matrix_norm_args_ord!(&str, isize, f64);

macro_rules! impl_vector_norm_args_ord {
    ($($t: ty),* $(,)?) => {$(
        impl TryFrom<$t> for VectorNormArgs {
            type Error = Error;
            fn try_from(ord: $t) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, ..Default::default() })
            }
        }

        impl TryFrom<($t, bool)> for VectorNormArgs {
            type Error = Error;
            fn try_from((ord, keepdims): ($t, bool)) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, keepdims, ..Default::default() })
            }
        }

        impl TryFrom<($t, AxesIndex<isize>)> for VectorNormArgs {
            type Error = Error;
            fn try_from((ord, axes): ($t, AxesIndex<isize>)) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, axes, keepdims: false })
            }
        }

        impl TryFrom<($t, AxesIndex<isize>, bool)> for VectorNormArgs {
            type Error = Error;
            fn try_from((ord, axes, keepdims): ($t, AxesIndex<isize>, bool)) -> Result<Self> {
                Ok(Self { ord: NormOrd::try_from(ord)?, axes, keepdims })
            }
        }
    )*};
}

impl_vector_norm_args_ord!(&str, isize, f64);

impl TryFrom<(NormOrd, AxesIndex<isize>)> for VectorNormArgs {
    type Error = Error;
    fn try_from((ord, axes): (NormOrd, AxesIndex<isize>)) -> Result<Self> {
        Ok(Self { ord, axes, keepdims: false })
    }
}

impl TryFrom<(NormOrd, AxesIndex<isize>, bool)> for VectorNormArgs {
    type Error = Error;
    fn try_from((ord, axes, keepdims): (NormOrd, AxesIndex<isize>, bool)) -> Result<Self> {
        Ok(Self { ord, axes, keepdims })
    }
}

impl FromStr for MatrixNormArgs {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s)
    }
}

impl FromStr for VectorNormArgs {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s)
    }
}

/* #endregion */

/* #region matrix_rank */

/// The relative tolerance handed to [`matrix_rank`] — the entry's argument type.
///
/// A scalar carries no device; an array does — so the two cannot share one
/// representation, and the array's device is checked against the input's before
/// the rank is computed. `None` selects the default `max(M, N) * eps`.
///
/// **Order notice.** The input's two matrix axes are its *leading* axes under a
/// row-major device but its *trailing* axes under a column-major one; the batch
/// is whatever remains, and an [`Array`](MatrixRankRtol::Array) tolerance is
/// broadcast against the batch in that same device order. On a row-major device
/// this is NumPy's trailing alignment against `shape(x)[:-2]`; on a
/// column-major device the matrix axes are the right-most and the tolerance
/// aligns from the opposite end.
///
/// `TR` is the input's real scalar type (`T::Real`) and `B` its device. `From`
/// converts the scalar, array, `()`, and `None` forms, so
/// `matrix_rank((x, 1e-8))` (an `f64` input), `matrix_rank((x, rtol_array))`,
/// and `matrix_rank((x, None))` all resolve. A scalar is taken as the input's
/// own real type — with an `f32` input write `1e-8_f32` (a bare `1e-8` is `f64`
/// and does not convert).
pub enum MatrixRankRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    /// The default tolerance `max(M, N) * eps`.
    None,
    /// One tolerance for every matrix — a scalar of the input's real type.
    Float(TR),
    /// One tolerance per matrix, broadcast against the batch in device order.
    Array(Tensor<TR, B, IxD>),
}

// not derived: `#[derive(Default)]` would add spurious `TR: Default, B: Default` bounds
#[allow(clippy::derivable_impls)]
impl<TR, B> Default for MatrixRankRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn default() -> Self {
        Self::None
    }
}

impl<TR, B> From<Tensor<TR, B, IxD>> for MatrixRankRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn from(rtol: Tensor<TR, B, IxD>) -> Self {
        Self::Array(rtol)
    }
}

// the scalar overload is written per real type, not as a generic `From<TR>`:
// a generic impl would overlap `From<()>`/`From<Option<()>>` at `TR = ()`
impl<B> From<f32> for MatrixRankRtol<f32, B>
where
    B: DeviceRawAPI<f32>,
{
    fn from(rtol: f32) -> Self {
        Self::Float(rtol)
    }
}

impl<B> From<f64> for MatrixRankRtol<f64, B>
where
    B: DeviceRawAPI<f64>,
{
    fn from(rtol: f64) -> Self {
        Self::Float(rtol)
    }
}

impl<TR, B> From<()> for MatrixRankRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    fn from(_: ()) -> Self {
        Self::None
    }
}

impl<TR, B> From<Option<()>> for MatrixRankRtol<TR, B>
where
    B: DeviceRawAPI<TR>,
{
    /// `None` is equivalent to `()`: both request the default tolerance.
    fn from(_: Option<()>) -> Self {
        Self::None
    }
}

/* #endregion */
