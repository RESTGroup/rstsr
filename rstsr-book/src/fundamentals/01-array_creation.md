# Array creation

## 1) 1-D tensor from rust vector

RSTSR tensor can be created by (owned) vector object.

In the following case, memory of vector object `vec` will be transferred to tensor object `tensor`.
Except for relatively small overhead (generating layout of tensor), **no explicit data copy occurs**.

```rust
{{#include ../../listings/fundamentals/listing-01-vec_to_tsr/src/main.rs}}
```

\* Note: This will generate tensor object for default CPU device. To 
