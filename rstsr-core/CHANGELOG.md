# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/RESTGroup/rstsr/compare/rstsr-core-v0.1.0...rstsr-core-v0.1.1) - 2025-02-28

### Other

- add flag to char
- change name TensorOrder to FlagOrder, previous name still usable
- simplify trait relations
- allow testing by manifests
- add fingerprint
- add alias
- add leading dimension associated methods to Ix2 tensor
- add c/f_contig/prefer to tensor associated methods
- from char (instead of try_into)
- add method to_prefer
- add char try_into and flip
- update rule for c/f-prefer, that last/first shape dim = 1 is considered as contiguous
- add enum TensorMutable
- add to_contig
- add uninitialized
