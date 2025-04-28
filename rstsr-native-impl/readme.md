# RSTSR rust native implementation to simple operators

This crate includes some native implementation (such as tensor addition, reduction, layout-change operations).

Only CPU (both serial and rayon parallel) are of concern.

It is splitted from `rstsr-core`, so to make this crate `rstsr-native-impl` more emphasis on computation, and this `rstsr-core` more emphasis on device and tensor structure/trait definition.

This crate can also be used if you wish to use tensor computation utilities, without actually defining a tensor object.
In another word, this crate is **device-free**. Only serial and rayon parallel is concerned in this crate.

THis crate uses utilities in `rstsr-common`, such as layout, broadcasting, iterator.
