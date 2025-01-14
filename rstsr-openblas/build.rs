use std::{error::Error, path::PathBuf};

/// Generate link search paths from a list of paths.
///
/// This allows paths like `/path/to/lib1:/path/to/lib2` to be split into
/// individual paths.
fn generate_link_search_paths(paths: &[Result<String, impl Error + Clone>]) -> Vec<String> {
    paths
        .iter()
        .map(|path| {
            path.clone()
                .unwrap_or_default()
                .split(":")
                .map(|path| path.to_string())
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>()
}

/// Check if the library is found in the given paths.
fn check_library_found(
    lib_name: &str,
    lib_paths: &[String],
    lib_extension: &[String],
) -> Option<String> {
    for path in lib_paths {
        for ext in lib_extension {
            let lib_path = PathBuf::from(&path).join(format!("lib{}.{}", lib_name, ext));
            if lib_path.exists() {
                return Some(lib_path.to_string_lossy().to_string());
            }
        }
    }
    return None;
}

fn main() {
    // following build is only for development
    // currently, the crate user is responsible to link openblas by themselves
    println!("cargo:rerun-if-env-changed=RSTSR_DEV");
    if std::env::var("RSTSR_DEV").is_ok() {
        let lib_paths = generate_link_search_paths(&[std::env::var("LD_LIBRARY_PATH")]);
        if let Some(path) = check_library_found("openblas", &lib_paths, &["so".to_string()]) {
            let path = std::fs::canonicalize(path).unwrap();
            let path = path.parent().unwrap().display();
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}
