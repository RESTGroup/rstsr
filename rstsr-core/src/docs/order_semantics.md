# Device default orders: [`RowMajor`] and [`ColMajor`]

RSTSR stores a tensor's data in a linear buffer, and a [`Layout`] (shape,
stride, offset) interprets that buffer as an n-dimensional object. The same
buffer admits two families of interpretation:

- **Row-major** (C-like, NumPy-like): the *last* axis varies fastest in
  memory; a freshly created tensor is C-contiguous.
- **Column-major** (F-like, Fortran/Julia-like): the *first* axis varies
  fastest in memory; a freshly created tensor is F-contiguous.

Which family is used by default is a property of the **device**, called the
*device default order*:

```rust
# use rstsr::prelude::*;
let mut device = DeviceCpu::default();
device.set_default_order(ColMajor); // or RowMajor; RowMajor is the default
                                   // under this crate's default features
```

This page explains what the default order affects. Functions also state their
own order behavior in their docstrings: order-independent ones carry the
one-line notice "This function behaves identically under [`RowMajor`] and
[`ColMajor`] device default orders", and order-dependent ones carry the
**Row/Column Major Notice** warning. (Functions documented before this
convention are being retrofitted.)

## What the default order controls

Shape-driven operations consult the device default order when the user does
not pin an order themselves:

| Situation | Row-major default | Column-major default |
|--|--|--|
| Creation with shape input ([`zeros`], [`ones`], [`empty`], [`full`], [`eye`], and [`asarray`] with shape) | C-contiguous result | F-contiguous result |
| [`reshape`] family, when a copy is required | copy into C-contiguous | copy into F-contiguous |
| Broadcasting (element-wise operators, [`assign`](crate::tensor::assignment::assign()), [`broadcast_to`], ...) | shapes align from the last axis (NumPy rule) | shapes align from the first axis (Fortran/Julia rule) |
| [`to_contig`] / [`to_prefer`] with the device default order passed as `order` | C-contiguous result | F-contiguous result |
| Axis iteration | row-major traversal | column-major traversal |

Layout-only manipulations ([`transpose`], slicing, [`flip`], ...) are not
affected at all. Element-wise computations are unaffected in their results;
only the memory arrangement of newly allocated results follows the default
order. For every other function, consult its own docstring notice rather than
assuming.

The rest of this page demonstrates the three cases worth understanding in
depth: creation, broadcasting, and reshape.

## Creation: same shape, different memory arrangement

Creating by shape under the two orders produces tensors with identical
logical shapes but different strides (the same linear sequence fills the
tensor; only the mapping between logical indices and memory positions
differs):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((6, &device)).into_shape([2, 3]);
println!("{a}");
// [[ 0 1 2]
//  [ 3 4 5]]
println!("{:?}", a.layout());
// 2-Dim (dyn), contiguous: Cc
// shape: [2, 3], stride: [3, 1], offset: 0
```

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(ColMajor);
let a = rt::arange((6, &device)).into_shape([2, 3]);
println!("{a}");
// [[ 0 2 4]
//  [ 1 3 5]]
println!("{:?}", a.layout());
// 2-Dim (dyn), contiguous: Ff
// shape: [2, 3], stride: [1, 2], offset: 0
```

One exception to keep in mind: the [`tensor_from_nested!`] macro always
produces a row-major (C-contiguous) layout, regardless of the device default
order.

## Broadcasting: alignment from the last or the first axis

Under [`RowMajor`], shapes align from the **last** axis - the NumPy rule.
Under [`ColMajor`], shapes align from the **first** axis - the
Fortran/Julia rule:

```rust
# use rstsr::prelude::*;
// A      (4d array):  8 x 1 x 6 x 1
// B      (3d array):      7 x 1 x 5
// row-major: aligned from the last axis
// ---------------------------------
// Result (4d array):  8 x 7 x 6 x 5
let shape1 = vec![8, 1, 6, 1];
let shape2 = vec![7, 1, 5];
let result = rt::broadcast_shapes(&[shape1, shape2], RowMajor);
println!("{result:?}");
// [8, 7, 6, 5]
# assert_eq!(result, vec![8, 7, 6, 5]);
```

```rust
# use rstsr::prelude::*;
// A      (4d array):  1 x 6 x 1 x 8
// B      (3d array):  5 x 1 x 7
// column-major: aligned from the first axis
// ---------------------------------
// Result (4d array):  5 x 6 x 7 x 8
let shape1 = vec![1, 6, 1, 8];
let shape2 = vec![5, 1, 7];
let result = rt::broadcast_shapes(&[shape1, shape2], ColMajor);
println!("{result:?}");
// [5, 6, 7, 8]
# assert_eq!(result, vec![5, 6, 7, 8]);
```

Code written NumPy-style usually keeps working under [`ColMajor`] only when
shapes are written Fortran-style (dimensions pre-padded on the other side).

## Reshape: iteration invariance

[`reshape`] reads the input in the logical order given by its order
(the device default order, unless specified) and writes the values into the
output in that same order: the i-th element in reading order of the input
becomes the i-th element in reading order of the output. Reshaping never
changes the values, nor their order in the reading sequence - only the shape,
and (when layouts are incompatible) the memory arrangement:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((6, &device)).into_shape([2, 3]);
let b = a.reshape([3, 2]);
println!("{}", a.reshape([6]));
// [ 0 1 2 3 4 5]
println!("{}", b.reshape([6]));
// [ 0 1 2 3 4 5]
# assert_eq!(format!("{}", a.reshape([6])), "[ 0 1 2 3 4 5]");
# assert_eq!(format!("{}", b.reshape([6])), "[ 0 1 2 3 4 5]");
```

Which reshapes avoid a copy depends on the order: for a tensor that is
C-contiguous under [`RowMajor`] (or F-contiguous under [`ColMajor`]), the
contiguous run of axes is the *trailing* axes in the row-major case and the
*leading* axes in the column-major case. In the notation `(4, [6, 9])` used
by [`reshape`], the bracketed axes form one contiguous run. Under
[`ColMajor`], the same slice that was `(4, [6, 9])`-contiguous in the
row-major discussion of [`reshape`] merges its leading dimensions for free,
while merging across the boundary requires a copy:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(ColMajor);
// contiguous situation: ([4, 6], 9); the first two dimensions are contiguous
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
println!("{:?}", a.layout());
// 3-Dim (dyn), contiguous: f
// shape: [4, 6, 9], stride: [1, 4, 32], offset: 0

// merging the leading (contiguous) dimensions needs no copy
assert!(!a.reshape([24, 9]).is_owned());
// merging across the contiguity boundary requires a copy
assert!(a.reshape([4, 54]).is_owned());
```

See [`reshape`] for the full discussion of when data cloning happens.
