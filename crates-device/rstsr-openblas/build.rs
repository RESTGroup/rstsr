//! Development-only build script for `rstsr-openblas`.
//!
//! When the `RSTSR_DEV` environment variable is set, this script wires up
//! dynamic linking against a system/conda OpenBLAS so that `cargo test` works
//! out of the box. It is a no-op for downstream users, who are expected to link
//! OpenBLAS themselves (and to leave `RSTSR_DEV` unset).
//!
//! What it does — only under `RSTSR_DEV`, and only when the `dynamic_loading`
//! feature is *not* enabled (that feature resolves the library at runtime via
//! `libloading` instead, so link-time wiring would be redundant and wrong):
//!
//! 1. Collects library search directories from the full set of path variables that matter across
//!    platforms — `LD_LIBRARY_PATH` (Linux loader), `DYLD_LIBRARY_PATH` (macOS loader),
//!    `LIBRARY_PATH` (build-time hint), `PATH` (Windows DLL dirs) and the project's `REST_EXT_DIR`
//!    — and emits `cargo:rustc-link-search` for each.
//! 2. Bakes an `-rpath` for the loader-style dirs so the test binary finds the dylibs at runtime
//!    without re-exporting the environment.
//! 3. Locates the OpenBLAS shared library. macOS conda ships only the versioned
//!    `libopenblas.0.dylib` (no unversioned `libopenblas.dylib`), which defeats `-lopenblas`; in
//!    that case we link the file by absolute path.
//! 4. Links the OpenMP runtime that matches the platform/toolchain: `omp` on macOS (LLVM, as used
//!    by clang and by conda's OpenBLAS), `gomp` on Linux (GNU). Override either with
//!    `RSTSR_OPENMP_LIB`. Only linked when the `openmp` feature is on, since that is the only cfg
//!    that references the `omp_*` symbols at link time.

use std::path::PathBuf;

/// Library-path environment variables, in priority order.
const LIB_PATH_VARS: &[&str] = &["REST_EXT_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBRARY_PATH", "PATH"];

/// Vars whose directories are also baked into `-rpath` (runtime lookup).
/// `LIBRARY_PATH`/`PATH` are build-time / executable hints, not loader paths,
/// so they are excluded from rpath.
const RPATH_VARS: &[&str] = &["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "REST_EXT_DIR"];

/// True only when we should perform link-time wiring.
fn is_dev_link() -> bool {
    std::env::var("RSTSR_DEV").is_ok() && !cfg!(feature = "dynamic_loading")
}

/// Collect existing directories from the given env vars, de-duplicated, order
/// preserved. Both `:` (unix) and `;` (windows) are accepted as separators so
/// the same logic works everywhere.
fn collect_dirs(vars: &[&str]) -> Vec<PathBuf> {
    let sep = if cfg!(windows) { ';' } else { ':' };
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for var in vars {
        let Ok(val) = std::env::var(var) else {
            continue;
        };
        for p in val.split(sep) {
            if p.is_empty() {
                continue;
            }
            let pb = PathBuf::from(p);
            if pb.is_dir() && seen.insert(pb.clone()) {
                out.push(pb);
            }
        }
    }
    out
}

/// Link OpenBLAS. Prefer the unversioned name (so `-lopenblas` resolves and
/// propagates to consumers); fall back to the versioned file by absolute path
/// (macOS conda only ships `libopenblas.0.dylib`).
fn link_openblas(dirs: &[PathBuf]) {
    let suffixes: &[&str] = if cfg!(target_os = "macos") {
        &[".dylib", ".a"]
    } else if cfg!(target_os = "windows") {
        &[".dll.a", ".lib"]
    } else {
        &[".so", ".a"]
    };

    // 1) unversioned libopenblas.<suffix> -> standard `-lopenblas`.
    for d in dirs {
        for s in suffixes {
            if d.join(format!("libopenblas{s}")).exists() {
                println!("cargo:rustc-link-lib=openblas");
                return;
            }
        }
    }

    // 2) versioned, e.g. libopenblas.0.dylib / libopenblas.so.0. Skip arch-tagged variants
    //    (libopenblasp-r0.3.33.dylib, libopenblas_armv8p-r0.3.33.dylib, ...) by requiring a digit right
    //    after `libopenblas.`.
    for d in dirs {
        let Ok(entries) = std::fs::read_dir(d) else {
            continue;
        };
        let mut found: Vec<PathBuf> = Vec::new();
        for ent in entries.flatten() {
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("libopenblas.") {
                continue;
            }
            if !suffixes.iter().any(|s| name.ends_with(s)) {
                continue;
            }
            let after = &name["libopenblas.".len()..];
            if after.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                if let Ok(p) = std::fs::canonicalize(ent.path()) {
                    found.push(p);
                }
            }
        }
        // pick the shortest name as the canonical SONAME (libopenblas.0.dylib)
        if let Some(p) =
            found.into_iter().min_by_key(|p| p.file_name().map(|n| n.to_string_lossy().len()).unwrap_or(usize::MAX))
        {
            println!("cargo:rustc-link-arg={}", p.display());
            return;
        }
    }

    // Nothing found — emit the bare name so the linker raises its usual error,
    // which is more actionable than a silent no-op.
    println!(
        "cargo:warning=RSTSR_DEV set but libopenblas was not found in \
         LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/LIBRARY_PATH/REST_EXT_DIR"
    );
    println!("cargo:rustc-link-lib=openblas");
}

/// Choose the OpenMP runtime library name for this platform.
fn omp_lib_name() -> String {
    if let Ok(v) = std::env::var("RSTSR_OPENMP_LIB") {
        if !v.is_empty() {
            return v;
        }
    }
    if cfg!(target_os = "macos") {
        "omp".into()
    } else {
        "gomp".into()
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=RSTSR_DEV");
    println!("cargo:rerun-if-env-changed=RSTSR_OPENMP_LIB");
    for v in LIB_PATH_VARS {
        println!("cargo:rerun-if-env-changed={v}");
    }

    if !is_dev_link() {
        return;
    }

    let dirs = collect_dirs(LIB_PATH_VARS);
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    // embed rpath so the test binary resolves the dylibs at runtime without
    // re-exporting DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
    for d in collect_dirs(RPATH_VARS) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }

    link_openblas(&dirs);

    // Only the `openmp` feature references `omp_*` symbols at link time
    // (via the FFI extern declarations); link the runtime then and only then.
    if cfg!(feature = "openmp") {
        let omp = omp_lib_name();
        println!("cargo:rustc-link-lib={omp}");
    }
}
