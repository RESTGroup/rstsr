extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{transmute, ManuallyDrop};

/* #region definitions */

#[derive(Debug, Clone)]
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

unsafe impl<C> Send for DataOwned<C> where C: Send {}
unsafe impl<C> Send for DataRef<'_, C> where C: Send {}
unsafe impl<C> Sync for DataRef<'_, C> where C: Sync {}
unsafe impl<C> Send for DataMut<'_, C> where C: Send {}
unsafe impl<C> Sync for DataCow<'_, C> where C: Sync {}
unsafe impl<C> Send for DataCow<'_, C> where C: Send {}
unsafe impl<C> Send for DataArc<C> where C: Send {}
unsafe impl<C> Sync for DataArc<C> where C: Sync {}

/* #endregion */

/* #region definitions not fully utilized */

#[derive(Debug)]
pub enum DataMutable<'a, S> {
    Owned(DataOwned<S>),
    RefMut(DataMut<'a, S>),
    ToBeCloned(DataRef<'a, S>, DataOwned<S>),
}

#[derive(Debug)]
pub enum DataReference<'a, S> {
    Ref(DataRef<'a, S>),
    RefMut(DataMut<'a, S>),
}

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

/* #endregion */

/* #region data traits */

pub trait DataAPI {
    type Data: Clone;
    fn raw(&self) -> &Self::Data;
    fn into_owned(self) -> DataOwned<Self::Data>;
    fn into_shared(self) -> DataArc<Self::Data>;
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef::from(self.raw())
    }
}

pub trait DataMutAPI: DataAPI {
    fn raw_mut(&mut self) -> &mut Self::Data;
    fn as_mut(&mut self) -> DataMut<Self::Data> {
        DataMut::TrueRef(self.raw_mut())
    }
}

pub trait DataOwnedAPI: DataMutAPI {}

pub trait DataForceMutAPI<C>: DataAPI<Data = C> {
    unsafe fn force_mut(&self) -> DataMut<'_, C>;
}

/* #endregion */

/* #region impl DataAPI */

impl<C> DataAPI for DataOwned<C>
where
    C: Clone,
{
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        &self.raw
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        DataArc::from(self.raw)
    }
}

impl<C> DataAPI for DataRef<'_, C>
where
    C: Clone,
{
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataRef::TrueRef(raw) => raw,
            DataRef::ManuallyDropOwned(raw) => raw,
        }
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataRef::TrueRef(raw) => DataOwned::from(raw.clone()),
            DataRef::ManuallyDropOwned(raw) => {
                let v = ManuallyDrop::into_inner(raw);
                DataOwned::from(v.clone())
            },
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataRef::TrueRef(raw) => DataArc::from(raw.clone()),
            DataRef::ManuallyDropOwned(raw) => {
                let v = ManuallyDrop::into_inner(raw);
                DataArc::from(v.clone())
            },
        }
    }
}

impl<C> DataAPI for DataMut<'_, C>
where
    C: Clone,
{
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataMut::TrueRef(raw) => raw,
            DataMut::ManuallyDropOwned(raw) => raw,
        }
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataMut::TrueRef(raw) => DataOwned::from(raw.clone()),
            DataMut::ManuallyDropOwned(raw) => {
                let v = ManuallyDrop::into_inner(raw);
                DataOwned::from(v.clone())
            },
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataMut::TrueRef(raw) => DataArc::from(raw.clone()),
            DataMut::ManuallyDropOwned(raw) => {
                let v = ManuallyDrop::into_inner(raw);
                DataArc::from(v.clone())
            },
        }
    }
}

impl<C> DataAPI for DataCow<'_, C>
where
    C: Clone,
{
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        match self {
            DataCow::Owned(data) => data.raw(),
            DataCow::Ref(data) => data.raw(),
        }
    }

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

impl<C> DataAPI for DataArc<C>
where
    C: Clone,
{
    type Data = C;

    #[inline]
    fn raw(&self) -> &Self::Data {
        &self.raw
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned::from(Arc::try_unwrap(self.raw).ok().unwrap())
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        self
    }
}

/* #endregion */

/* #region impl DataMutAPI */

impl<C> DataMutAPI for DataOwned<C>
where
    C: Clone,
{
    #[inline]
    fn raw_mut(&mut self) -> &mut Self::Data {
        &mut self.raw
    }
}

impl<C> DataMutAPI for DataMut<'_, C>
where
    C: Clone,
{
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

impl<T> DataForceMutAPI<Vec<T>> for DataRef<'_, Vec<T>>
where
    T: Clone,
{
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

macro_rules! impl_data_force_vec {
    ($data: ty) => {
        impl<T> DataForceMutAPI<Vec<T>> for $data
        where
            T: Clone,
        {
            unsafe fn force_mut(&self) -> DataMut<'_, Vec<T>> {
                transmute(self.as_ref().force_mut())
            }
        }
    };
}

impl_data_force_vec!(DataOwned<Vec<T>>);
impl_data_force_vec!(DataMut<'_, Vec<T>>);
impl_data_force_vec!(DataCow<'_, Vec<T>>);
impl_data_force_vec!(DataArc<Vec<T>>);

/* #endregion */

/* #region DataCow */

pub trait DataIntoCowAPI<'a>: DataAPI {
    fn into_cow(self) -> DataCow<'a, Self::Data>;
}

impl<'a, C> DataIntoCowAPI<'a> for DataOwned<C>
where
    C: Clone,
{
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        DataCow::Owned(self)
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataRef<'a, C>
where
    C: Clone,
{
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        DataCow::Ref(self)
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataMut<'a, C>
where
    C: Clone,
{
    #[inline]
    fn into_cow(self) -> DataCow<'a, C> {
        match self {
            DataMut::TrueRef(data) => DataRef::from(&*data).into_cow(),
            DataMut::ManuallyDropOwned(data) => DataRef::from_manually_drop(data).into_cow(),
        }
    }
}

impl<'a, C> DataIntoCowAPI<'a> for DataCow<'a, C>
where
    C: Clone,
{
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
