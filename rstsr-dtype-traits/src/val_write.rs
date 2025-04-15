use std::mem::MaybeUninit;

/// Sets the value of the current object to `val`.
///
/// This trait function is designed for unifying `T` and `MaybeUninit<T>` for
/// our purposes.
///
/// Note that for `MaybeUninit<T>`, if `T` is a type with non trivial `Drop`,
/// then memory could leak for using this function.
pub trait ValWriteAPI<T> {
    fn write(&mut self, val: T) -> &mut T;
}

impl<T> ValWriteAPI<T> for T {
    #[inline(always)]
    fn write(&mut self, val: T) -> &mut T {
        *self = val;
        self
    }
}

impl<T> ValWriteAPI<T> for MaybeUninit<T> {
    #[inline(always)]
    fn write(&mut self, val: T) -> &mut T {
        self.write(val)
    }
}

#[test]
fn playground() {
    fn inner<W, T>(a: &mut W, b: T)
    where
        W: ValWriteAPI<T>,
    {
        a.write(b);
    }

    let mut a = 0.0;
    inner(&mut a, 1.0);
    println!("a {:?}", a);

    let mut b = MaybeUninit::uninit();
    inner(&mut b, 1.0);
    println!("b {:?}", b);
}
