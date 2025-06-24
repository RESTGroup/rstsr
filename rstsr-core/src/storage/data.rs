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

unsafe impl<C> Send for DataOwned<C> where C: Send {}
unsafe impl<C> Send for DataRef<'_, C> where C: Send {}
unsafe impl<C> Sync for DataRef<'_, C> where C: Sync {}
unsafe impl<C> Send for DataMut<'_, C> where C: Send {}
unsafe impl<C> Sync for DataCow<'_, C> where C: Sync {}
unsafe impl<C> Send for DataCow<'_, C> where C: Send {}
unsafe impl<C> Send for DataArc<C> where C: Send {}
unsafe impl<C> Sync for DataArc<C> where C: Sync {}
unsafe impl<C> Send for DataReference<'_, C> where C: Send {}
unsafe impl<C> Sync for DataReference<'_, C> where C: Sync {}

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

impl<'a, C> From<&'a mut C> for DataMut<'a, C> {
    #[inline]
    fn from(data: &'a mut C) -> Self {
        DataMut::TrueRef(data)
    }
}

impl<C> DataMut<'_, C> {
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
