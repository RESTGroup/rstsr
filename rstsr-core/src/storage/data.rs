extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{transmute, ManuallyDrop};
use duplicate::duplicate_item;

/* #region definitions */

#[derive(Debug)]
pub struct DataOwned<C> {
    pub(crate) raw: C,
}

#[derive(Debug)]
pub enum DataRef<'a, C> {
    TrueRef(&'a C),
    ManuallyDropOwned(ManuallyDrop<C>),
}

#[derive(Debug)]
pub enum DataMut<'a, C> {
    TrueRef(&'a mut C),
    ManuallyDropOwned(ManuallyDrop<C>),
}

#[derive(Debug)]
pub enum DataCow<'a, C> {
    Owned(DataOwned<C>),
    Ref(DataRef<'a, C>),
}

#[derive(Debug)]
pub struct DataArc<C> {
    pub(crate) raw: Arc<C>,
}

#[derive(Debug)]
pub enum DataReference<'a, C> {
    Ref(DataRef<'a, C>),
    Mut(DataMut<'a, C>),
}

// Send/Sync below are tighter than a plain derive would suggest:
// - a variant holding `&'a C` (`DataRef::TrueRef`) is `Send` only when `C: Sync`,
// - `Arc<C>` is `Send`/`Sync` only when `C: Send + Sync`,
// so every impl that can expose a shared reference (directly or through
// `Arc`/`DataRef`/`DataCow`/`DataReference`) requires `C: Send + Sync`.
// The previous looser bounds (`C: Send` only) allowed e.g.
// `DataRef<'_, Cell<i32>>` to be sent across threads — a data race.
unsafe impl<C> Send for DataOwned<C> where C: Send {}
unsafe impl<C> Send for DataRef<'_, C> where C: Send + Sync {}
unsafe impl<C> Sync for DataRef<'_, C> where C: Sync {}
unsafe impl<C> Send for DataMut<'_, C> where C: Send {}
unsafe impl<C> Sync for DataCow<'_, C> where C: Sync {}
unsafe impl<C> Send for DataCow<'_, C> where C: Send + Sync {}
unsafe impl<C> Send for DataArc<C> where C: Send + Sync {}
unsafe impl<C> Sync for DataArc<C> where C: Send + Sync {}
unsafe impl<C> Send for DataReference<'_, C> where C: Send + Sync {}
unsafe impl<C> Sync for DataReference<'_, C> where C: Send + Sync {}

/* #endregion */

/* #region specific implementations */

impl<C> From<C> for DataOwned<C> {
    #[inline]
    fn from(data: C) -> Self {
        Self { raw: data }
    }
}

impl<C> DataOwned<C> {
    #[inline]
    pub fn into_raw(self) -> C {
        self.raw
    }
}

impl<'a, C> From<&'a C> for DataRef<'a, C> {
    #[inline]
    fn from(data: &'a C) -> Self {
        DataRef::TrueRef(data)
    }
}

impl<C> DataRef<'_, C> {
    /// Wraps data that this wrapper will own without managing.
    ///
    /// # Contract
    ///
    /// The wrapped data must remain valid for the lifetime `'a` declared on
    /// the resulting [`DataRef`]. Construction sites rely on this: e.g.
    /// `asarray` fabricates a `Vec` (via `from_raw_parts`) over a borrowed
    /// slice, whose buffer stays valid for `'a`. The [`ManuallyDrop`]
    /// wrapper guarantees the data is never freed here.
    #[inline]
    pub fn from_manually_drop(data: ManuallyDrop<C>) -> Self {
        DataRef::ManuallyDropOwned(data)
    }

    #[inline]
    pub fn is_true_ref(&self) -> bool {
        matches!(self, DataRef::TrueRef(_))
    }

    #[inline]
    pub fn is_manually_drop_owned(&self) -> bool {
        matches!(self, DataRef::ManuallyDropOwned(_))
    }
}

impl<'a, C> DataRef<'a, C> {
    /// Returns the wrapped reference carrying the declared lifetime `'a`, or
    /// `None` for a [`DataRef::ManuallyDropOwned`] wrapper.
    ///
    /// Only a true reference can be handed out persistently: for the
    /// ManuallyDrop-owned variant the data lives inside the wrapper value
    /// itself, so no `'a`-valid reference to it exists. Use
    /// [`DataRef::as_slice_ref`] for the common `Vec`-buffer case.
    #[inline]
    pub fn try_as_true_ref(&self) -> Option<&'a C> {
        match self {
            DataRef::TrueRef(r) => Some(*r),
            DataRef::ManuallyDropOwned(_) => None,
        }
    }
}

impl<'a, T> DataRef<'a, Vec<T>> {
    /// Returns the slice over the underlying buffer, carrying `'a`.
    ///
    /// Sound for both variants: [`DataRef::TrueRef`] borrows the owner's
    /// buffer directly; a [`DataRef::ManuallyDropOwned`] `Vec` is fabricated
    /// over a buffer that stays valid for `'a` (the [`DataRef::from_manually_drop`]
    /// contract), and the slice is re-derived from its pointer and length
    /// while `self` is still alive. The wrapper value itself may be dropped
    /// later — only the (never-freed) buffer is referenced.
    #[inline]
    pub fn as_slice_ref(&self) -> &'a [T] {
        match self {
            DataRef::TrueRef(r) => &r[..],
            DataRef::ManuallyDropOwned(md) => {
                let ptr = md.as_ptr();
                let len = md.len();
                // SAFETY: the Vec was fabricated over a buffer valid for 'a
                // (from_manually_drop contract); rebuilding the slice from its
                // pointer and length is sound and does not drop the Vec.
                unsafe { core::slice::from_raw_parts(ptr, len) }
            },
        }
    }
}

impl<'a, C> From<&'a mut C> for DataMut<'a, C> {
    #[inline]
    fn from(data: &'a mut C) -> Self {
        DataMut::TrueRef(data)
    }
}

impl<C> DataMut<'_, C> {
    /// Wraps data that this wrapper will own without managing.
    ///
    /// # Contract
    ///
    /// The wrapped data must remain valid *and exclusively borrowable* for
    /// the lifetime `'a` declared on the resulting [`DataMut`] (the mutable
    /// counterpart of the [`DataRef::from_manually_drop`] contract). The
    /// [`ManuallyDrop`] wrapper guarantees the data is never freed here.
    #[inline]
    pub fn from_manually_drop(data: ManuallyDrop<C>) -> Self {
        DataMut::ManuallyDropOwned(data)
    }

    #[inline]
    pub fn is_true_ref(&self) -> bool {
        matches!(self, DataMut::TrueRef(_))
    }

    #[inline]
    pub fn is_manually_drop_owned(&self) -> bool {
        matches!(self, DataMut::ManuallyDropOwned(_))
    }
}

impl<'a, C> DataMut<'a, C> {
    /// Consumes the wrapper and returns the exclusive reference carrying the
    /// declared lifetime `'a`, or `None` for a [`DataMut::ManuallyDropOwned`]
    /// wrapper.
    ///
    /// The receiver is consuming because a `&'a mut` cannot be reborrowed
    /// out. Returning `None` (instead of laundering) keeps this persistently
    /// sound: ManuallyDrop-owned data has no `'a`-valid exclusive reference
    /// at the `C` level. Use [`DataMut::into_slice_mut`] for the `Vec`-buffer
    /// case.
    #[inline]
    pub fn try_into_true_mut(self) -> Option<&'a mut C> {
        match self {
            DataMut::TrueRef(m) => Some(m),
            DataMut::ManuallyDropOwned(_) => None,
        }
    }
}

impl<'a, T> DataMut<'a, Vec<T>> {
    /// Consumes the wrapper and returns the mutable slice over the underlying
    /// buffer, carrying `'a`.
    ///
    /// Sound for both variants (see [`DataRef::as_slice_ref`]): a
    /// [`DataMut::ManuallyDropOwned`] `Vec` is fabricated over an
    /// exclusively-borrowable, `'a`-valid buffer, and consuming `self`
    /// transfers that exclusivity to the returned slice. The `Vec` itself is
    /// never dropped.
    #[inline]
    pub fn into_slice_mut(self) -> &'a mut [T] {
        match self {
            DataMut::TrueRef(m) => &mut m[..],
            DataMut::ManuallyDropOwned(mut md) => {
                let ptr = md.as_mut_ptr();
                let len = md.len();
                // SAFETY: same contract as DataRef::as_slice_ref, exclusive:
                // the fabricated Vec covers an 'a-valid, exclusively
                // borrowable buffer; the Vec is never dropped.
                unsafe { core::slice::from_raw_parts_mut(ptr, len) }
            },
        }
    }
}

impl<C> DataCow<'_, C> {
    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, DataCow::Owned(_))
    }

    #[inline]
    pub fn is_ref(&self) -> bool {
        matches!(self, DataCow::Ref(_))
    }
}

impl<C> From<Arc<C>> for DataArc<C> {
    #[inline]
    fn from(data: Arc<C>) -> Self {
        Self { raw: data }
    }
}

impl<C> From<C> for DataArc<C> {
    #[inline]
    fn from(data: C) -> Self {
        Self { raw: Arc::new(data) }
    }
}

impl<C> DataArc<C> {
    #[inline]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.raw)
    }

    #[inline]
    pub fn weak_count(&self) -> usize {
        Arc::weak_count(&self.raw)
    }
}

impl<C> DataReference<'_, C> {
    #[inline]
    pub fn is_ref(&self) -> bool {
        matches!(self, DataReference::Ref(_))
    }

    #[inline]
    pub fn is_mut(&self) -> bool {
        matches!(self, DataReference::Mut(_))
    }
}

/* #endregion */

/* #region data traits */

pub trait DataAPI {
    type Data;
    fn raw(&self) -> &Self::Data;
    fn as_ref(&'_ self) -> DataRef<'_, Self::Data> {
        DataRef::from(self.raw())
    }
}

pub trait DataCloneAPI
where
    Self: DataAPI,
    Self::Data: Clone,
{
    fn into_owned(self) -> DataOwned<Self::Data>;
    fn into_shared(self) -> DataArc<Self::Data>;
}

pub trait DataMutAPI: DataAPI {
    fn raw_mut(&mut self) -> &mut Self::Data;
    fn as_mut(&'_ mut self) -> DataMut<'_, Self::Data> {
        DataMut::TrueRef(self.raw_mut())
    }
}

pub trait DataOwnedAPI: DataMutAPI {}

pub trait DataForceMutAPI<C>: DataAPI<Data = C> {
    /// # Safety
    ///
    /// This function is highly unsafe, as it entirely bypasses Rust's lifetime
    /// and borrowing rules.
    unsafe fn force_mut(&self) -> DataMut<'_, C>;
}

/* #endregion */

/* #region impl DataCloneAPI */

impl<C> DataAPI for DataOwned<C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        &self.raw
    }
}

impl<C> DataCloneAPI for DataOwned<C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        DataArc::from(self.raw)
    }
}

impl<C> DataAPI for DataRef<'_, C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataRef::TrueRef(raw) => raw,
            DataRef::ManuallyDropOwned(raw) => raw,
        }
    }
}

impl<C> DataCloneAPI for DataRef<'_, C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataRef::TrueRef(raw) => DataOwned::from(raw.clone()),
            DataRef::ManuallyDropOwned(raw) => DataOwned::from(ManuallyDrop::into_inner(raw.clone())),
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataRef::TrueRef(raw) => DataArc::from(raw.clone()),
            DataRef::ManuallyDropOwned(raw) => DataArc::from(ManuallyDrop::into_inner(raw.clone())),
        }
    }
}

impl<C> DataAPI for DataMut<'_, C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataMut::TrueRef(raw) => raw,
            DataMut::ManuallyDropOwned(raw) => raw,
        }
    }
}

impl<C> DataCloneAPI for DataMut<'_, C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataMut::TrueRef(raw) => DataOwned::from(raw.clone()),
            DataMut::ManuallyDropOwned(raw) => DataOwned::from(ManuallyDrop::into_inner(raw.clone())),
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataMut::TrueRef(raw) => DataArc::from(raw.clone()),
            DataMut::ManuallyDropOwned(raw) => DataArc::from(ManuallyDrop::into_inner(raw.clone())),
        }
    }
}

impl<C> DataAPI for DataCow<'_, C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataCow::Owned(data) => data.raw(),
            DataCow::Ref(data) => data.raw(),
        }
    }
}

impl<C> DataCloneAPI for DataCow<'_, C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataCow::Owned(data) => data,
            DataCow::Ref(data) => data.into_owned(),
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataCow::Owned(data) => DataArc::from(data.into_raw()),
            DataCow::Ref(data) => data.into_shared(),
        }
    }
}

impl<C> DataAPI for DataArc<C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        &self.raw
    }
}

impl<C> DataCloneAPI for DataArc<C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned::from(Arc::try_unwrap(self.raw).ok().unwrap())
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        self
    }
}

impl<C> DataAPI for DataReference<'_, C> {
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataReference::Ref(data) => data.raw(),
            DataReference::Mut(data) => data.raw(),
        }
    }
}

impl<C> DataCloneAPI for DataReference<'_, C>
where
    C: Clone,
{
    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataReference::Ref(data) => data.into_owned(),
            DataReference::Mut(data) => data.into_owned(),
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataReference::Ref(data) => data.into_shared(),
            DataReference::Mut(data) => data.into_shared(),
        }
    }
}

/* #endregion */

/* #region impl DataMutAPI */

impl<C> DataMutAPI for DataOwned<C> {
    #[inline]
    fn raw_mut(&mut self) -> &mut Self::Data {
        &mut self.raw
    }
}

impl<C> DataMutAPI for DataMut<'_, C> {
    #[inline]
    fn raw_mut(&mut self) -> &mut Self::Data {
        match self {
            DataMut::TrueRef(raw) => raw,
            DataMut::ManuallyDropOwned(raw) => raw,
        }
    }
}

impl<C> DataMutAPI for DataArc<C>
where
    C: Clone,
{
    #[inline]
    fn raw_mut(&mut self) -> &mut Self::Data {
        Arc::make_mut(&mut self.raw)
    }
}

/* #endregion */

/* #region impl DataForceMutAPI */

impl<T> DataForceMutAPI<Vec<T>> for DataRef<'_, Vec<T>> {
    unsafe fn force_mut(&self) -> DataMut<'_, Vec<T>> {
        let (ptr, len) = match self {
            DataRef::TrueRef(raw) => (raw.as_ptr(), raw.len()),
            DataRef::ManuallyDropOwned(raw) => (raw.as_ptr(), raw.len()),
        };
        let vec = unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) };
        let vec = ManuallyDrop::new(vec);
        DataMut::ManuallyDropOwned(vec)
    }
}

#[duplicate_item(
    Data;
    [DataOwned<Vec<T>>];
    [DataMut<'_, Vec<T>>];
    [DataCow<'_, Vec<T>>];
    [DataArc<Vec<T>>];
    [DataReference<'_, Vec<T>>];
)]
impl<T> DataForceMutAPI<Vec<T>> for Data {
    unsafe fn force_mut(&self) -> DataMut<'_, Vec<T>> {
        transmute(self.as_ref().force_mut())
    }
}

/* #endregion */

/* #region DataCow */

pub trait DataIntoCowAPI<'a>
where
    Self: DataAPI,
{
    fn into_cow(self) -> DataCow<'a, Self::Data>;
}

impl<'a, C> DataIntoCowAPI<'a> for DataOwned<C> {
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        DataCow::Owned(self)
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataRef<'a, C> {
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        DataCow::Ref(self)
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataMut<'a, C> {
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        match self {
            DataMut::TrueRef(data) => DataRef::from(&*data).into_cow(),
            DataMut::ManuallyDropOwned(data) => DataRef::from_manually_drop(data).into_cow(),
        }
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataCow<'a, C> {
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        self
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataArc<C>
where
    C: Clone,
{
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        DataCow::Owned(self.into_owned())
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataReference<'a, C> {
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        match self {
            DataReference::Ref(data) => data.into_cow(),
            DataReference::Mut(data) => data.into_cow(),
        }
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_trait_data() {
        let vec = vec![10, 20, 30];
        println!("===");
        println!("{:?}", vec.as_ptr());
        let data = DataOwned { raw: vec.clone() };
        let data_ref = data.as_ref();
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.raw().as_ptr());
        println!("{:?}", data_ref_ref.raw().as_ptr());
        let data_ref2 = data_ref.into_owned();
        println!("{:?}", data_ref2.raw().as_ptr());

        println!("===");
        let data_ref = DataRef::from_manually_drop(ManuallyDrop::new(vec.clone()));
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.raw().as_ptr());
        println!("{:?}", data_ref_ref.raw().as_ptr());
        let mut data_ref2 = data_ref.into_owned();
        println!("{:?}", data_ref2.raw().as_ptr());
        data_ref2.raw_mut()[1] = 10;
    }
}
