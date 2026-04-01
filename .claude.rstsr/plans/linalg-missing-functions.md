---
name: linalg-missing-functions
description: Missing linear algebra functions to implement in rstsr-linalg-traits
type: project
---

# Linear Algebra Functions Implementation Plan

Comparison with `scipy.linalg` (primary reference) and `numpy.linalg` (complementary).

## Current Implementation Status

| Function | Status | LAPACK Driver | Notes |
|----------|--------|---------------|-------|
| `cholesky` | ✅ Done | POTRF | |
| `det` | ✅ Done | LU-based | |
| `eigh` | ✅ Done | SYEVD/HEEVD | Symmetric/Hermitian eigen |
| `eigvalsh` | ✅ Done | SYEVD/HEEVD | Symmetric/Hermitian eigenvalues only |
| `inv` | ✅ Done | GETRF+GETRI | |
| `pinv` | ✅ Done | SVD-based | |
| `qr` | ✅ Done | GEQRF/GEQP3 | With pivoting support |
| `slogdet` | ✅ Done | LU-based | |
| `solve_general` | ✅ Done | GESV | |
| `solve_symmetric` | ✅ Done | SYSV/HESV | |
| `solve_triangular` | ✅ Done | TRTRS | |
| `svd` | ✅ Done | GESVD/GESDD | |
| `svdvals` | ✅ Done | GESVD/GESDD | |

## Priority 1: Core Functions (High Impact)

### General Eigenvalue Problem
- `eig`, `eigvals` - General matrix eigenvalues/eigenvectors
- **LAPACK**: GEEV (real/complex)
- **Why**: Fundamental for many applications, currently only symmetric supported
- **Approach**: Add trait `EigAPI`, implement via LAPACK GEEV drivers

### LU Decomposition
- `lu`, `lu_factor`, `lu_solve` - LU decomposition with pivoting
- **LAPACK**: GETRF, GETRS
- **Why**: Foundation for many solvers, useful for determinant calculation
- **Approach**: Add traits `LuAPI`, `LuFactorAPI`, `LuSolveAPI`

### Least Squares Solver
- `lstsq` - Linear least squares problem
- **LAPACK**: GELS, GELSD (SVD-based), GELSY (QR-based)
- **Why**: Essential for overdetermined systems
- **Approach**: Add trait `LstsqAPI`

### Matrix/Vector Norms
- `norm` - Matrix and vector norms (Frobenius, 1-norm, 2-norm, inf-norm, nuclear)
- **LAPACK**: LANGE for matrix norms
- **Why**: Basic utility function, needed for condition number
- **Approach**: Add trait `NormAPI`

### Condition Number
- `cond` - Matrix condition number
- **LAPACK**: GECON (LU-based), POCON (Cholesky-based)
- **Why**: Important for numerical stability analysis
- **Approach**: Add trait `CondAPI`, depends on norm or LU

### Matrix Rank
- `matrix_rank` - Numerical matrix rank
- **LAPACK**: Uses SVD thresholding
- **Why**: Basic matrix property
- **Approach**: Implement based on SVD with tolerance

## Priority 2: Important Decompositions

### Schur Decomposition
- `schur`, `rsf2csf` - Schur decomposition, real-to-complex conversion
- **LAPACK**: GEES, GEESX
- **Why**: Used for matrix functions (expm, logm), QZ decomposition
- **Approach**: Add trait `SchurAPI`

### LDL Decomposition
- `ldl` - LDL^T decomposition for symmetric indefinite matrices
- **LAPACK**: SYTRF, SYTRF2, SYTRI2
- **Why**: Handles indefinite matrices where Cholesky fails
- **Approach**: Add trait `LdlAPI`

### Hessenberg Form
- `hessenberg` - Reduce matrix to Hessenberg form
- **LAPACK**: GEHRD, ORGHR
- **Why**: Preprocessing for eigenvalue algorithms, QR algorithm
- **Approach**: Add trait `HessenbergAPI`

### Polar Decomposition
- `polar` - Polar decomposition U*P
- **LAPACK**: Uses SVD (no dedicated driver)
- **Why**: Useful in optimization and control
- **Approach**: Implement based on existing SVD

### QZ Decomposition (Generalized Schur)
- `qz`, `ordqz` - QZ decomposition for matrix pairs
- **LAPACK**: GGES, GGES3, TGSEN
- **Why**: Generalized eigenvalue problems AX = λBX
- **Approach**: Add trait `QzAPI`

## Priority 3: Matrix Functions

### Matrix Exponential
- `expm` - Matrix exponential
- **LAPACK**: Uses Schur + Padé approximation
- **Why**: Differential equations, control theory
- **Approach**: Requires Schur decomposition first

### Matrix Logarithm
- `logm` - Matrix logarithm
- **LAPACK**: Uses Schur + inverse scaling
- **Why**: Structure preserving transformations
- **Approach**: Requires Schur decomposition

### Matrix Square Root
- `sqrtm` - Matrix square root
- **LAPACK**: Uses Schur decomposition
- **Why**: Polar decomposition variant
- **Approach**: Requires Schur decomposition

### Other Matrix Functions
- `cosm`, `sinm`, `tanm`, `coshm`, `sinhm`, `tanhm`, `signm`
- `funm` - Arbitrary matrix function
- `fractional_matrix_power` - A^t for any real t
- **Why**: Advanced functionality, can be built on Schur

## Priority 4: Matrix Equation Solvers

### Sylvester Equation
- `solve_sylvester` - Solve AX + XB = Q
- **LAPACK**: TRSYL
- **Why**: Control theory, model reduction
- **Approach**: Requires Schur decomposition

### Lyapunov Equations
- `solve_continuous_lyapunov` - A*X + X*A^H = Q
- `solve_discrete_lyapunov` - A*X*A^H - X = -Q
- **LAPACK**: TRSYL for continuous, DTRSYL-like for discrete
- **Why**: Stability analysis in control systems
- **Approach**: Uses Schur decomposition

### Riccati Equations
- `solve_continuous_are`, `solve_discrete_are`
- **LAPACK**: No direct driver, uses multiple LAPACK routines
- **Why**: Optimal control, filtering
- **Approach**: Complex implementation using Schur/QZ

## Priority 5: Banded/Specialized Functions

### Banded Matrix Operations
- `solve_banded`, `solveh_banded` - Banded system solvers
- `cholesky_banded` - Cholesky for banded matrices
- `eig_banded`, `eigvals_banded` - Eigen for banded matrices
- **LAPACK**: GBTRF, GBTRS, PBTRF, PBTRS, SBEVD, etc.
- **Why**: Efficiency for structured matrices

### Tridiagonal Eigen
- `eigh_tridiagonal`, `eigvalsh_tridiagonal`
- **LAPACK**: STEDC, STERF
- **Why**: Specialized for quantum chemistry, physics
- **Note**: Already partially discussed in faer_impl

## Priority 6: Utility and Special Matrices

### Utility Functions
- `bandwidth` - Matrix bandwidth
- `issymmetric`, `ishermitian` - Check matrix properties
- `matrix_balance` - Balance matrix for eigenvalue computation
- `subspace_angles` - Principal angles between subspaces
- `orthogonal_procrustes` - Procrustes problem

### Special Matrix Generators
- `toeplitz`, `circulant`, `hankel`, `hadamard`, `hilbert`, `pascal`, `dft`, etc.
- **Why**: Test matrices, structured matrices

## Implementation Order Recommendation

1. **Phase 1**: `norm`, `cond`, `matrix_rank` (utilities)
2. **Phase 2**: `eig`, `eigvals` (general eigenvalue)
3. **Phase 3**: `lu`, `lu_factor`, `lu_solve` (LU decomposition)
4. **Phase 4**: `lstsq` (least squares)
5. **Phase 5**: `schur` (Schur decomposition)
6. **Phase 6**: Matrix functions (`expm`, `logm`, `sqrtm`)
7. **Phase 7**: `ldl`, `hessenberg`, `polar`
8. **Phase 8**: Matrix equation solvers
9. **Phase 9**: Banded matrix functions
10. **Phase 10**: Special matrices and utilities

## Implementation Notes

- All traits should follow existing pattern: `XxxAPI`, `xxx_f`, `xxx`
- Use `derive_builder` for argument structs
- Support both LAPACK drivers (via `driver` parameter) and faer backend
- Add result structs similar to `EighResult`, `SVDResult`, `QRResult`
- Implement tests for real (f64) and complex (c64) types
- Consider row-major vs column-major layout handling