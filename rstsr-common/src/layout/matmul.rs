/*!

Layout manuplication for matmul and other linalg operations

# Rules for matmul

We Refer to [Python array API](https://data-apis.org/array-api/2024.12/specification/generated/array_api.matmul.html) for more information.

The rules below are written for row-major; the last two axes of each operand
are the matmul dimensions and any leading axes broadcast.

| Id | A | B | C |
|----|---|---|---|
| 1. | `        N` | `        N` | `         ` |
| 2. | `     M, K` | `     K, N` | `     M, N` |
| 3. | `        K` | `..., K, N` | `   ..., N` |
| 4. | `..., M, K` | `        K` | `   ..., M` |
| 5. | `     M, K` | `..., K, N` | `..., M, N` |
| 6. | `..., M, K` | `     K, N` | `..., M, N` |
| 7. | `..., M, K` | `..., K, N` | `..., M, N` |

For col-major, the same rules apply *with all axes reversed*: the matmul
dimensions are the first two of each operand and any trailing axes broadcast.
This is implemented by delegating to the row-major routine on reversed-and-
swapped inputs, using the identity
`C[t, m, n] = A[t, m, k] @ B[t, k, n]` (row-major) `==`
`C[n, m, t] = B[n, k, t] @ A[k, m, t]` (col-major).

*/

use crate::prelude_dev::*;

/// Rules of matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulType {
    InnerDot,
    GEMM22,
    GEVM,
    GEMV,
    GEMM2X,
    GEMMX2,
    GEMMXX,
}

#[derive(Clone, Debug)]
pub struct LayoutMatMulConfig<DA, DB>
where
    DA: DimAPI,
    DB: DimAPI,
    Self: LayoutMatMulAPI<DA, DB>,
{
    pub matmul_type: MatMulType,
    pub lc: Layout<<Self as LayoutMatMulAPI<DA, DB>>::DC>,
    pub la_rest: Option<Layout<IxD>>,
    pub lb_rest: Option<Layout<IxD>>,
    pub lc_rest: Option<Layout<IxD>>,
    pub la_matmul: Layout<IxD>,
    pub lb_matmul: Layout<IxD>,
    pub lc_matmul: Layout<IxD>,
}

pub trait LayoutMatMulAPI<DA, DB>
where
    DA: DimAPI,
    DB: DimAPI,
    Self: Sized,
{
    type DC: DimAPI;
    /// Layout configuration for matmul.
    ///
    /// For order, currently we only accept deterministic order.
    fn layout_matmul(la: &Layout<DA>, lb: &Layout<DB>, order: FlagOrder) -> Result<Self>;
}

// rule 1
impl LayoutMatMulAPI<Ix1, Ix1> for LayoutMatMulConfig<Ix1, Ix1> {
    type DC = Ix0;
    fn layout_matmul(la: &Layout<Ix1>, lb: &Layout<Ix1>, _: FlagOrder) -> Result<Self> {
        // check shape
        rstsr_assert_eq!(la.shape(), lb.shape(), InvalidLayout)?;
        let lc = unsafe { Layout::new_unchecked([], [], 0) };
        Ok(LayoutMatMulConfig {
            matmul_type: MatMulType::InnerDot,
            lc: lc.clone(),
            la_rest: None,
            lb_rest: None,
            lc_rest: None,
            la_matmul: la.to_dim()?,
            lb_matmul: lb.to_dim()?,
            lc_matmul: lc.to_dim()?,
        })
    }
}

// rule 2
impl LayoutMatMulAPI<Ix2, Ix2> for LayoutMatMulConfig<Ix2, Ix2> {
    type DC = Ix2;
    fn layout_matmul(la: &Layout<Ix2>, lb: &Layout<Ix2>, order: FlagOrder) -> Result<Self> {
        // check and generate shape
        rstsr_assert_eq!(la.shape()[1], lb.shape()[0], InvalidLayout)?;
        let sc = [la.shape()[0], lb.shape()[1]];
        // layout order determination
        let lc = match order {
            RowMajor => sc.c(),
            ColMajor => sc.f(),
        };
        // return layout configuration
        Ok(LayoutMatMulConfig {
            matmul_type: MatMulType::GEMM22,
            lc: lc.clone(),
            la_rest: None,
            lb_rest: None,
            lc_rest: None,
            la_matmul: la.to_dim()?,
            lb_matmul: lb.to_dim()?,
            lc_matmul: lc.to_dim()?,
        })
    }
}

fn layout_matmul_dyn_row_major(la: &Layout<IxD>, lb: &Layout<IxD>) -> Result<LayoutMatMulConfig<IxD, IxD>> {
    let na = la.ndim();
    let nb = lb.ndim();
    match (na, nb) {
        (1, 1) => {
            // rule 1: vector inner dot
            rstsr_assert_eq!(la.shape(), lb.shape(), InvalidLayout)?;
            let lc = unsafe { Layout::new_unchecked(vec![], vec![], 0) };
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::InnerDot,
                lc: lc.clone(),
                la_rest: None,
                lb_rest: None,
                lc_rest: None,
                la_matmul: la.to_dim()?,
                lb_matmul: lb.to_dim()?,
                lc_matmul: lc.to_dim()?,
            })
        },
        (2, 2) => {
            // rule 2: matrix multiplication
            // check and generate shape
            rstsr_assert_eq!(la.shape()[1], lb.shape()[0], InvalidLayout)?;
            let sc = vec![la.shape()[0], lb.shape()[1]];
            // layout order determination
            let lc = sc.c();
            // return layout configuration
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMM22,
                lc: lc.clone(),
                la_rest: None,
                lb_rest: None,
                lc_rest: None,
                la_matmul: la.to_dim()?,
                lb_matmul: lb.to_dim()?,
                lc_matmul: lc.to_dim()?,
            })
        },
        (1, 2..) => {
            // rule 3: | `        K` | `..., K, N` | `   ..., N` |
            // check and generate shape
            let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
            rstsr_assert_eq!(la.shape()[0], lb_matmul.shape()[0], InvalidLayout)?;
            // layout order determination
            let mut sc = lb_rest.shape().clone();
            sc.push(lb_matmul.shape()[1]);
            let lc = sc.c();
            // return layout configuration
            let (lc_rest, lc_matmul) = lc.dim_split_at(-1)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEVM,
                lc: lc.to_dim()?,
                la_rest: None,
                lb_rest: Some(lb_rest),
                lc_rest: Some(lc_rest),
                la_matmul: la.to_dim()?,
                lb_matmul: lb_matmul.to_dim()?,
                lc_matmul: lc_matmul.to_dim()?,
            })
        },
        (2.., 1) => {
            // rule 4: | `..., M, K` | `        K` | `   ..., M` |
            // check and generate shape
            let (la_rest, la_matmul) = la.dim_split_at(-2)?;
            rstsr_assert_eq!(lb.shape()[0], la_matmul.shape()[1], InvalidLayout)?;
            // layout order determination
            let mut sc = la_rest.shape().clone();
            sc.push(la_matmul.shape()[0]);
            let lc = sc.c();
            // return layout configuration
            let (lc_rest, lc_matmul) = lc.dim_split_at(-1)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMV,
                lc: lc.to_dim()?,
                la_rest: Some(la_rest),
                lb_rest: None,
                lc_rest: Some(lc_rest),
                la_matmul: la_matmul.to_dim()?,
                lb_matmul: lb.to_dim()?,
                lc_matmul: lc_matmul.to_dim()?,
            })
        },
        (2, 3..) => {
            // rule 5: | `     M, K` | `..., K, N` | `..., M, N` |
            // check and generate shape
            let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
            rstsr_assert_eq!(la.shape()[1], lb_matmul.shape()[0], InvalidLayout)?;
            // layout order determination
            let mut sc = lb_rest.shape().clone();
            sc.append(&mut vec![la.shape()[0], lb_matmul.shape()[1]]);
            let lc = sc.c();
            // return layout configuration
            let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMM2X,
                lc: lc.to_dim()?,
                la_rest: None,
                lb_rest: Some(lb_rest),
                lc_rest: Some(lc_rest),
                la_matmul: la.to_dim()?,
                lb_matmul: lb_matmul.to_dim()?,
                lc_matmul: lc_matmul.to_dim()?,
            })
        },
        (3.., 2) => {
            // rule 6: | `..., M, K` | `     K, N` | `..., M, N` |
            // check and generate shape
            let (la_rest, la_matmul) = la.dim_split_at(-2)?;
            rstsr_assert_eq!(la_matmul.shape()[1], lb.shape()[0], InvalidLayout)?;
            // layout order determination
            let mut sc = la_rest.shape().clone();
            sc.append(&mut vec![la_matmul.shape()[0], lb.shape()[1]]);
            let lc = sc.c();
            // return layout configuration
            let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMMX2,
                lc: lc.to_dim()?,
                la_rest: Some(la_rest),
                lb_rest: None,
                lc_rest: Some(lc_rest),
                la_matmul: la_matmul.to_dim()?,
                lb_matmul: lb.to_dim()?,
                lc_matmul: lc_matmul.to_dim()?,
            })
        },
        (3.., 3..) => {
            // check and generate shape
            let (la_rest, la_matmul) = la.dim_split_at(-2)?;
            let (lb_rest, lb_matmul) = lb.dim_split_at(-2)?;
            rstsr_assert_eq!(la_matmul.shape()[1], lb_matmul.shape()[0], InvalidLayout)?;
            let (la_rest_b, lb_rest_b) = broadcast_layout(&la_rest, &lb_rest, RowMajor)?;
            // layout order determination
            let mut sc = la_rest_b.shape().clone();
            sc.append(&mut vec![la_matmul.shape()[0], lb_matmul.shape()[1]]);
            let lc = sc.c();
            // return layout configuration
            let (lc_rest, lc_matmul) = lc.dim_split_at(-2)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMMXX,
                lc: lc.to_dim()?,
                la_rest: Some(la_rest_b),
                lb_rest: Some(lb_rest_b),
                lc_rest: Some(lc_rest),
                la_matmul: la.to_dim()?,
                lb_matmul: lb_matmul.to_dim()?,
                lc_matmul: lc_matmul.to_dim()?,
            })
        },
        (0, _) | (_, 0) => rstsr_invalid!((na, nb), "In matmul, 0-dim is not allowed."),
    }
}

/// Resolve matmul layouts against a caller-provided `lc`.
///
/// Unlike [`layout_matmul_dyn_row_major`], which *constructs* `lc` from `la`
/// and `lb`, this function takes the real `lc` as the source of truth for the
/// batch shape (it comes from the caller and may be strided / non-zero-offset)
/// and derives `la_rest` / `lb_rest` by broadcasting them **to** `lc_rest`.
/// All returned layouts are split from the real `la` / `lb` / `lc`, so their
/// strides and offsets are valid for indexing into the actual operand buffers.
///
/// This is the canonical entry point used by the `DeviceMatMulAPI`
/// implementations (faer, BLAS backends, naive CPU); it centralizes the rule
/// table and the batch-broadcasting so the device drivers only have to dispatch
/// on [`MatMulType`] and iterate the rest layouts.
pub fn layout_matmul_dyn_row_major_with_lc(
    la: &Layout<IxD>,
    lb: &Layout<IxD>,
    lc: &Layout<IxD>,
) -> Result<LayoutMatMulConfig<IxD, IxD>> {
    let na = la.ndim();
    let nb = lb.ndim();
    let nc = lc.ndim();
    match (na, nb) {
        (1, 1) => {
            // rule 1: vector inner dot
            rstsr_assert_eq!(la.shape(), lb.shape(), InvalidLayout)?;
            rstsr_assert_eq!(nc, 0, InvalidLayout)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::InnerDot,
                lc: lc.clone(),
                la_rest: None,
                lb_rest: None,
                lc_rest: None,
                la_matmul: la.clone(),
                lb_matmul: lb.clone(),
                lc_matmul: lc.clone(),
            })
        },
        (2, 2) => {
            // rule 2: matrix multiplication
            rstsr_assert_eq!(nc, 2, InvalidLayout)?;
            rstsr_assert_eq!(la.shape()[1], lb.shape()[0], InvalidLayout)?;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMM22,
                lc: lc.clone(),
                la_rest: None,
                lb_rest: None,
                lc_rest: None,
                la_matmul: la.clone(),
                lb_matmul: lb.clone(),
                lc_matmul: lc.clone(),
            })
        },
        (1, 2..) => {
            // rule 3: | `        K` | `..., K, N` | `   ..., N` |
            rstsr_assert_eq!(nb, nc + 1, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-1)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-1)?;
            let la_rest = broadcast_layout_to_first(&lc_r, &la_r, RowMajor)?.1;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEVM,
                lc: lc.clone(),
                la_rest: Some(la_rest),
                lb_rest: Some(lb_r),
                lc_rest: Some(lc_r),
                la_matmul: la_m.dim_insert(0)?,
                lb_matmul: lb_m,
                lc_matmul: lc_m.dim_insert(0)?,
            })
        },
        (2.., 1) => {
            // rule 4: | `..., M, K` | `        K` | `   ..., M` |
            rstsr_assert_eq!(na, nc + 1, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-1)?;
            let (lc_r, lc_m) = lc.dim_split_at(-1)?;
            let lb_rest = broadcast_layout_to_first(&lc_r, &lb_r, RowMajor)?.1;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMV,
                lc: lc.clone(),
                la_rest: Some(la_r),
                lb_rest: Some(lb_rest),
                lc_rest: Some(lc_r),
                la_matmul: la_m,
                lb_matmul: lb_m.dim_insert(1)?,
                lc_matmul: lc_m.dim_insert(1)?,
            })
        },
        (2, 3..) => {
            // rule 5: | `     M, K` | `..., K, N` | `..., M, N` |
            rstsr_assert_eq!(nb, nc, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            let la_rest = broadcast_layout_to_first(&lc_r, &la_r, RowMajor)?.1;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMM2X,
                lc: lc.clone(),
                la_rest: Some(la_rest),
                lb_rest: Some(lb_r),
                lc_rest: Some(lc_r),
                la_matmul: la_m,
                lb_matmul: lb_m,
                lc_matmul: lc_m,
            })
        },
        (3.., 2) => {
            // rule 6: | `..., M, K` | `     K, N` | `..., M, N` |
            rstsr_assert_eq!(na, nc, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            let lb_rest = broadcast_layout_to_first(&lc_r, &lb_r, RowMajor)?.1;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMMX2,
                lc: lc.clone(),
                la_rest: Some(la_r),
                lb_rest: Some(lb_rest),
                lc_rest: Some(lc_r),
                la_matmul: la_m,
                lb_matmul: lb_m,
                lc_matmul: lc_m,
            })
        },
        (3.., 3..) => {
            // rule 7: | `..., M, K` | `..., K, N` | `..., M, N` |
            rstsr_assert_eq!(na, nc, InvalidLayout)?;
            rstsr_assert_eq!(nb, nc, InvalidLayout)?;
            let (la_r, la_m) = la.dim_split_at(-2)?;
            let (lb_r, lb_m) = lb.dim_split_at(-2)?;
            let (lc_r, lc_m) = lc.dim_split_at(-2)?;
            // both A and B batch dims broadcast against C's batch dims
            let la_rest = broadcast_layout_to_first(&lc_r, &la_r, RowMajor)?.1;
            let lb_rest = broadcast_layout_to_first(&lc_r, &lb_r, RowMajor)?.1;
            Ok(LayoutMatMulConfig {
                matmul_type: MatMulType::GEMMXX,
                lc: lc.clone(),
                la_rest: Some(la_rest),
                lb_rest: Some(lb_rest),
                lc_rest: Some(lc_r),
                la_matmul: la_m,
                lb_matmul: lb_m,
                lc_matmul: lc_m,
            })
        },
        (0, _) | (_, 0) => rstsr_invalid!((na, nb), "In matmul, 0-dim is not allowed."),
    }
}

fn layout_matmul_dyn_col_major(la: &Layout<IxD>, lb: &Layout<IxD>) -> Result<LayoutMatMulConfig<IxD, IxD>> {
    // For col-major, we re-use the row-major implementation via the identity
    //     C[t, m, n] = A[t, m, k] @ B[t, k, n]   (row-major)
    // <=> C[n, m, t] = B[n, k, t] @ A[k, m, t]   (col-major)
    // i.e. reverse all axes and swap A/B. So we delegate to
    // `layout_matmul_dyn_row_major(lb_rev, la_rev)` and then reverse-axes (and
    // swap A/B back) on every layout field that the row-major impl returns.
    //
    // Note that rules 5/6/7 (broadcasting matmul) are only supported by the
    // row-major rules in the array API spec; for col-major we accept them
    // here, but the corresponding `DeviceMatMulAPI` impl must follow the same
    // reverse-axes-and-swap convention (see e.g. `device_faer/matmul.rs`).
    let na = la.ndim();
    let nb = lb.ndim();
    if na == 0 || nb == 0 {
        return rstsr_invalid!((na, nb), "In matmul, 0-dim is not allowed.");
    }
    let la_rev = la.reverse_axes();
    let lb_rev = lb.reverse_axes();
    let cfg = layout_matmul_dyn_row_major(&lb_rev, &la_rev)?;
    Ok(LayoutMatMulConfig {
        matmul_type: cfg.matmul_type,
        lc: cfg.lc.reverse_axes(),
        // row-major's `la_*` corresponds to (reversed) B, so it maps back to
        // col-major's `lb_*` (after reversing axes again).
        la_rest: cfg.lb_rest.map(|l| l.reverse_axes()),
        lb_rest: cfg.la_rest.map(|l| l.reverse_axes()),
        lc_rest: cfg.lc_rest.map(|l| l.reverse_axes()),
        la_matmul: cfg.lb_matmul.reverse_axes(),
        lb_matmul: cfg.la_matmul.reverse_axes(),
        lc_matmul: cfg.lc_matmul.reverse_axes(),
    })
}

impl LayoutMatMulAPI<IxD, IxD> for LayoutMatMulConfig<IxD, IxD> {
    type DC = IxD;
    fn layout_matmul(la: &Layout<IxD>, lb: &Layout<IxD>, order: FlagOrder) -> Result<Self> {
        match order {
            RowMajor => layout_matmul_dyn_row_major(la, lb),
            ColMajor => layout_matmul_dyn_col_major(la, lb),
        }
    }
}

macro_rules! impl_fixed {
    ($DA:ident, $DB:ident, $DC:ident) => {
        impl LayoutMatMulAPI<$DA, $DB> for LayoutMatMulConfig<$DA, $DB> {
            type DC = $DC;
            fn layout_matmul(la: &Layout<$DA>, lb: &Layout<$DB>, order: FlagOrder) -> Result<Self> {
                let la = la.to_dim::<IxD>()?;
                let lb = lb.to_dim::<IxD>()?;
                let cfg = LayoutMatMulConfig::layout_matmul(&la, &lb, order)?;
                return Ok(LayoutMatMulConfig {
                    matmul_type: cfg.matmul_type,
                    lc: cfg.lc.into_dim()?,
                    la_rest: cfg.la_rest,
                    lb_rest: cfg.lb_rest,
                    lc_rest: cfg.lc_rest,
                    la_matmul: cfg.la_matmul,
                    lb_matmul: cfg.lb_matmul,
                    lc_matmul: cfg.lc_matmul,
                });
            }
        }
    };
}

// rule 3
impl_fixed!(Ix2, Ix1, Ix1);
impl_fixed!(Ix3, Ix1, Ix2);
impl_fixed!(Ix4, Ix1, Ix3);
impl_fixed!(Ix5, Ix1, Ix4);
impl_fixed!(Ix6, Ix1, Ix5);
impl_fixed!(Ix7, Ix1, Ix6);
impl_fixed!(Ix8, Ix1, Ix7);
impl_fixed!(Ix9, Ix1, Ix8);

// rule 4
impl_fixed!(Ix1, Ix2, Ix1);
impl_fixed!(Ix1, Ix3, Ix2);
impl_fixed!(Ix1, Ix4, Ix3);
impl_fixed!(Ix1, Ix5, Ix4);
impl_fixed!(Ix1, Ix6, Ix5);
impl_fixed!(Ix1, Ix7, Ix6);
impl_fixed!(Ix1, Ix8, Ix7);
impl_fixed!(Ix1, Ix9, Ix8);

// rule 5
impl_fixed!(Ix3, Ix2, Ix3);
impl_fixed!(Ix4, Ix2, Ix4);
impl_fixed!(Ix5, Ix2, Ix5);
impl_fixed!(Ix6, Ix2, Ix6);
impl_fixed!(Ix7, Ix2, Ix7);
impl_fixed!(Ix8, Ix2, Ix8);
impl_fixed!(Ix9, Ix2, Ix9);

// rule 6
impl_fixed!(Ix2, Ix3, Ix3);
impl_fixed!(Ix2, Ix4, Ix4);
impl_fixed!(Ix2, Ix5, Ix5);
impl_fixed!(Ix2, Ix6, Ix6);
impl_fixed!(Ix2, Ix7, Ix7);
impl_fixed!(Ix2, Ix8, Ix8);
impl_fixed!(Ix2, Ix9, Ix9);

// rule 7
impl_fixed!(Ix3, Ix3, Ix3);
impl_fixed!(Ix4, Ix4, Ix4);
impl_fixed!(Ix5, Ix5, Ix5);
impl_fixed!(Ix6, Ix6, Ix6);
impl_fixed!(Ix7, Ix7, Ix7);
impl_fixed!(Ix8, Ix8, Ix8);
impl_fixed!(Ix9, Ix9, Ix9);

// partial fixed
impl_fixed!(Ix1, IxD, IxD);
impl_fixed!(Ix2, IxD, IxD);
impl_fixed!(Ix3, IxD, IxD);
impl_fixed!(Ix4, IxD, IxD);
impl_fixed!(Ix5, IxD, IxD);
impl_fixed!(Ix6, IxD, IxD);
impl_fixed!(Ix7, IxD, IxD);
impl_fixed!(Ix8, IxD, IxD);
impl_fixed!(Ix9, IxD, IxD);

impl_fixed!(IxD, Ix1, IxD);
impl_fixed!(IxD, Ix2, IxD);
impl_fixed!(IxD, Ix3, IxD);
impl_fixed!(IxD, Ix4, IxD);
impl_fixed!(IxD, Ix5, IxD);
impl_fixed!(IxD, Ix6, IxD);
impl_fixed!(IxD, Ix7, IxD);
impl_fixed!(IxD, Ix8, IxD);
impl_fixed!(IxD, Ix9, IxD);

#[cfg(test)]
mod test_fixed {

    #[test]
    fn test_layout_matmul() {
        use super::*;
        let la = [4].c();
        let lb = [4].c();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.matmul_type, MatMulType::InnerDot);
        assert_eq!(config.lc.shape(), &[]);
        assert_eq!(config.la_matmul.shape(), &[4]);
        assert_eq!(config.lb_matmul.shape(), &[4]);

        let la = [5].c();
        let lb = [3, 4, 5, 6].f().swapaxes(0, 1).unwrap();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [4, 3, 6].c());

        let la = [3, 4, 5, 6].f().swapaxes(0, 1).unwrap();
        let lb = [6].c();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [4, 3, 5].c());

        let la = [7, 6].c();
        let lb = [2, 3, 4, 5, 6].f().swapaxes(-1, -2).unwrap();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [2, 3, 4, 7, 5].c());

        let la = [2, 3, 4, 5, 6].f().swapaxes(-1, -2).unwrap();
        let lb = [5, 7].c();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [2, 3, 4, 6, 7].c());

        let la = [4, 1, 2, 5, 6].f().swapaxes(0, 2).unwrap();
        let lb = [4, 3, 1, 6, 7].f().swapaxes(0, 2).unwrap();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [2, 3, 4, 5, 7].c());

        let la = [4, 3, 2, 5, 6].f().swapaxes(0, 2).unwrap();
        let lb = [4, 3, 2, 6, 7].f().swapaxes(0, 2).unwrap();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, RowMajor).unwrap();
        assert_eq!(config.lc, [2, 3, 4, 5, 7].c());

        // col-major broadcasting (mirror of the row-major cases above; the
        // matmul dims are the first two, the trailing dims broadcast).
        let la = [5, 6].f();
        let lb = [6, 7].f();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, ColMajor).unwrap();
        assert_eq!(config.lc, [5, 7].f());

        // rule 3 mirrored: K @ (K, N, ...) -> (N, ...)
        let la = [5].f();
        let lb = [5, 6, 3, 4].f();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, ColMajor).unwrap();
        assert_eq!(config.lc, [6, 3, 4].f());

        // rule 4 mirrored: (M, K, ...) @ K -> (M, ...)
        let la = [5, 6, 3, 4].f();
        let lb = [6].f();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, ColMajor).unwrap();
        assert_eq!(config.lc, [5, 3, 4].f());

        // rule 7 mirrored: full 5D x 5D batched matmul, including broadcast
        // on the trailing dims (`1` broadcasts against `2`/`3`).
        let la = [5, 6, 2, 1, 4].f();
        let lb = [6, 7, 1, 3, 4].f();
        let config = LayoutMatMulConfig::layout_matmul(&la, &lb, ColMajor).unwrap();
        assert_eq!(config.lc, [5, 7, 2, 3, 4].f());
    }
}
