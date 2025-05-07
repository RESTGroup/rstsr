use std::path::PathBuf;

fn main() {
    // following build is only for development
    // currently, the crate user is responsible to link openblas by themselves
    println!("cargo:rerun-if-env-changed=RSTSR_DEV");
    if std::env::var("RSTSR_DEV").is_ok() {
        std::env::var("LD_LIBRARY_PATH")
            .unwrap()
            .split(":")
            .filter(|path| !path.is_empty())
            .filter(|path| PathBuf::from(path).exists())
            .for_each(|path| {
                println!("cargo:rustc-link-search=native={path}");
            });
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-lib=gomp");
    }
}
