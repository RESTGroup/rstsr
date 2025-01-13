# rstsr Tensor

Basic structure of this crate:
```output
struct Tensor<R, D>
       |             \
struct Layout<D>        struct Data***<'l, S>
       |                trait  DataAPI            \
struct Ix<N>, IxD       trait  DataMutAPI            struct Storage<T, B>
trait  DimAPI           abbr   R (Representation)    trait  StorageAPI     \
abbr   D (Dimension)           |                     abbr   S (Storage)       type   Raw
                        lifetime  'l                        |                 abbr   T (Data Type)
                        ownership DataOwned          trait  DeviceAPI
                                  DataRef            abbr   B (Backend)
                                  DataMutRef
                                  ...
```

API documents for sub-modules or important classes:
- [`Layout`]