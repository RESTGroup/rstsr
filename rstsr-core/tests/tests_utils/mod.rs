//! Utilities of tests.

pub mod equality;

pub use equality::*;

/// Test configuration.
pub struct TestCfg<B> {
    /// Device instance (commonly used) to run the test on.
    pub device: B,

    /// Test cases to skip.
    pub skip: Vec<Vec<&'static str>>,

    /// Test to run. If None, all tests are allowed to run.
    pub allow: Option<Vec<Vec<&'static str>>>,

    /// Verbose mode. If true, print the skipped test cases.
    pub verbose: bool,
}

impl<B> TestCfg<B> {
    pub fn init(device: B, skip: Vec<&'static str>, allow: Option<Vec<&'static str>>) -> Self {
        // split skip and allow by "::"
        let skip = skip.into_iter().map(|s| s.split("::").collect()).collect();
        let allow = allow.map(|allow| allow.into_iter().map(|s| s.split("::").collect()).collect());
        Self { device, skip, allow, verbose: true }
    }

    /// Check if the test case with the given name should be skipped.
    pub fn to_test(&self, test_name: &[&str]) -> bool {
        // If any sequence in skip is a contiguous subsequence of test_name, skip the test.
        // For example, test_name `A::B::C::D`, then `B::C` will skip the test, but `A::C` will not.
        if self.skip.iter().any(|skip| test_name.windows(skip.len()).any(|w| w == skip)) {
            if self.verbose {
                eprintln!("skip test {}", test_name.join("::"));
            }
            return false;
        }
        // Only the sequence that starts with the prefix of test_name is allowed.
        // For example, test_name `A::B::C::D`, then `A::B` will allow the test, but `B::C` will not.
        if let Some(allow) = &self.allow {
            if !allow.iter().any(|allow| test_name.starts_with(allow)) {
                if self.verbose {
                    eprintln!("skip test {}", test_name.join("::"));
                }
                return false;
            }
        }
        true
    }
}

#[macro_export]
macro_rules! specify_test {
    ($item:literal) => {{
        static ITEM: &str = stringify!($item);
        if !TESTCFG.to_test(&[CATEGORY, FUNC, ITEM]) {
            eprintln!("skip test {CATEGORY}::{FUNC}::{ITEM}");
            return;
        }
    }};
}
