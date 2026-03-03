fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }


    #[cfg(not(target_os = "macos"))]
    {
        panic!("'accelerate' feature is only available for macOS target.");
    }
}
