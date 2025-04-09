# RSTSR Common

This crate includes some core light-weighted utilities used in RSTSR. Most computationally non-intensive operations are defined or declared in this crate.

This crate also involves iterator to layouts in CPU. These iterators should be fairly efficient (does not leverage SIMD optimization, so not fully efficient), and those iterators can be applied to other rust programs that only uses CPU as backend.

This crate becomes independent after RSTSR v0.2.6+.
