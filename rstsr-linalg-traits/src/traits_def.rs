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
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QRAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
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
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QRAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
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
   [PinvAPI           ] [pinv            ] [pinv_f            ];
   [QRAPI             ] [qr              ] [qr_f              ];
   [SLogDetAPI        ] [slogdet         ] [slogdet_f         ];
   [SolveGeneralAPI   ] [solve_general   ] [solve_general_f   ];
   [SolveSymmetricAPI ] [solve_symmetric ] [solve_symmetric_f ];
   [SolveTriangularAPI] [solve_triangular] [solve_triangular_f];
   [SVDAPI            ] [svd             ] [svd_f             ];
   [SVDvalsAPI        ] [svdvals         ] [svdvals_f         ];
)]
pub fn func<Args, Inp>(args: Args) -> <Args as LinalgAPI<Inp>>::Out
where
    Args: LinalgAPI<Inp>,
{
    Args::func(args)
}

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

/* #region eig */

/// Result structure for general eigenvalue decomposition (eig)
///
/// For non-symmetric/general matrices, eigenvalues can be complex even for real input.
/// This result structure always returns complex eigenvalues.
///
/// For real input matrices with complex eigenvalues, eigenvectors are also complex.
pub struct EigResult<W, VL, VR> {
    /// Eigenvalues (always complex for general matrices)
    pub eigenvalues: W,
    /// Left eigenvectors (optional)
    pub left_eigenvectors: Option<VL>,
    /// Right eigenvectors (optional)
    pub right_eigenvectors: Option<VR>,
}

impl<W, VL, VR> From<(W, Option<VL>, Option<VR>)> for EigResult<W, VL, VR> {
    fn from((w, vl, vr): (W, Option<VL>, Option<VR>)) -> Self {
        Self { eigenvalues: w, left_eigenvectors: vl, right_eigenvectors: vr }
    }
}

impl<W, VL, VR> From<EigResult<W, VL, VR>> for (W, Option<VL>, Option<VR>) {
    fn from(eig_result: EigResult<W, VL, VR>) -> Self {
        (eig_result.eigenvalues, eig_result.left_eigenvectors, eig_result.right_eigenvectors)
    }
}

/// Convert EigResult to (w, vr) tuple for default case (right eigenvectors only)
impl<W, VR> From<EigResult<W, W, VR>> for (W, VR) {
    fn from(eig_result: EigResult<W, W, VR>) -> Self {
        (eig_result.eigenvalues, eig_result.right_eigenvectors.unwrap())
    }
}

/// Convert EigResult to (w, vl, vr) tuple for both eigenvectors
impl<W, VL, VR> From<EigResult<W, VL, VR>> for (W, VL, VR) {
    fn from(eig_result: EigResult<W, VL, VR>) -> Self {
        (eig_result.eigenvalues, eig_result.left_eigenvectors.unwrap(), eig_result.right_eigenvectors.unwrap())
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", no_std, build_fn(error = "Error"))]
pub struct EigArgs_<'a, 'b, B, T>
where
    T: BlasFloat,
    B: DeviceAPI<T>,
{
    #[builder(setter(into))]
    pub a: TensorReference<'a, T, B, Ix2>,
    #[builder(setter(into, strip_option), default = "None")]
    pub b: Option<TensorReference<'b, T, B, Ix2>>,

    #[builder(setter(into), default = false)]
    pub left: bool,
    #[builder(setter(into), default = true)]
    pub right: bool,
}

pub type EigArgs<'a, 'b, B, T> = EigArgs_Builder<'a, 'b, B, T>;

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

/* #endregion */

/* #region slogdet */

pub struct SLogDetResult<T>
where
    T: BlasFloat,
{
    pub sign: T,
    pub logabsdet: T::Real,
}

impl<T> From<(T, T::Real)> for SLogDetResult<T>
where
    T: BlasFloat,
{
    fn from((sign, logabsdet): (T, T::Real)) -> Self {
        Self { sign, logabsdet }
    }
}

impl<T> From<SLogDetResult<T>> for (T, T::Real)
where
    T: BlasFloat,
{
    fn from(slogdet_result: SLogDetResult<T>) -> Self {
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
    T: BlasFloat,
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

/// Unified QR decomposition result for all modes
///
/// Different modes populate different fields:
/// - "reduced" (or "economic"): q=Some(Q), r=Some(R), p=Some/None
/// - "complete": q=Some(Q), r=Some(R), p=Some/None
/// - "r": q=None, r=Some(R), p=Some/None
/// - "raw": q=None, r=None, h=Some(H), tau=Some(tau), p=Some/None
///
/// Note: "economic" is an alias for "reduced" (SciPy vs NumPy terminology).
pub struct QRResult<Q, R, H, Tau, P> {
    /// Orthogonal matrix Q (Some for reduced/complete modes)
    pub q: Option<Q>,
    /// Upper triangular R (Some for reduced/complete/r modes)
    pub r: Option<R>,
    /// Packed Householder matrix H (Some for raw mode)
    pub h: Option<H>,
    /// Tau vector (Some for raw mode)
    pub tau: Option<Tau>,
    /// Pivot indices (Some if pivoting enabled)
    pub p: Option<P>,
}

impl<Q, R, H, Tau, P> From<(Option<Q>, Option<R>, Option<H>, Option<Tau>, Option<P>)> for QRResult<Q, R, H, Tau, P> {
    fn from((q, r, h, tau, p): (Option<Q>, Option<R>, Option<H>, Option<Tau>, Option<P>)) -> Self {
        Self { q, r, h, tau, p }
    }
}

impl<Q, R, H, Tau, P> From<QRResult<Q, R, H, Tau, P>> for (Option<Q>, Option<R>, Option<H>, Option<Tau>, Option<P>) {
    fn from(qr_result: QRResult<Q, R, H, Tau, P>) -> Self {
        (qr_result.q, qr_result.r, qr_result.h, qr_result.tau, qr_result.p)
    }
}

/// Convert QRResult to (Q, R) tuple for reduced/complete modes
/// Panics if q or r is None (i.e., for 'r' or 'raw' modes)
impl<Q, R, H, Tau, P> From<QRResult<Q, R, H, Tau, P>> for (Q, R) {
    fn from(qr_result: QRResult<Q, R, H, Tau, P>) -> Self {
        (qr_result.q.unwrap(), qr_result.r.unwrap())
    }
}

/// Convert QRResult to (Q, R, P) tuple for reduced/complete modes with pivoting
/// Panics if q, r, or p is None
impl<Q, R, H, Tau, P> From<QRResult<Q, R, H, Tau, P>> for (Q, R, P) {
    fn from(qr_result: QRResult<Q, R, H, Tau, P>) -> Self {
        (qr_result.q.unwrap(), qr_result.r.unwrap(), qr_result.p.unwrap())
    }
}

/* #endregion */
