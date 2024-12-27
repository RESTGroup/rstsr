extern crate alloc;

use alloc::sync::Arc;
use core::mem::ManuallyDrop;

#[derive(Debug, Clone)]
pub struct DataOwned<S>
where
    S: Sized,
{
    pub(crate) storage: S,
}

#[derive(Debug, Clone)]
pub enum DataRef<'a, S> {
    TrueRef(&'a S),
    ManuallyDropOwned(ManuallyDrop<S>),
}

#[derive(Debug)]
pub enum DataMut<'a, S> {
    TrueRef(&'a mut S),
    ManuallyDropOwned(ManuallyDrop<S>),
}

#[derive(Debug)]
pub enum DataCow<'a, S> {
    Owned(DataOwned<S>),
    Ref(DataRef<'a, S>),
}

#[derive(Debug)]
pub struct DataArc<S> {
    pub(crate) storage: Arc<S>,
}

unsafe impl<S> Send for DataArc<S> where S: Send {}
unsafe impl<S> Sync for DataArc<S> where S: Sync {}

impl<S> DataArc<S> {
    #[inline]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.storage)
    }

    #[inline]
    pub fn weak_count(&self) -> usize {
        Arc::weak_count(&self.storage)
    }
}

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

impl<S> From<S> for DataOwned<S> {
    #[inline]
    fn from(data: S) -> Self {
        Self { storage: data }
    }
}

impl<S> DataOwned<S> {
    #[inline]
    pub fn into_storage(self) -> S {
        self.storage
    }
}

impl<'a, S> From<&'a S> for DataRef<'a, S> {
    #[inline]
    fn from(data: &'a S) -> Self {
        DataRef::TrueRef(data)
    }
}

impl<S> DataRef<'_, S> {
    #[inline]
    pub fn from_manually_drop(data: ManuallyDrop<S>) -> Self {
        DataRef::ManuallyDropOwned(data)
    }
}

impl<'a, S> From<&'a mut S> for DataMut<'a, S> {
    #[inline]
    fn from(data: &'a mut S) -> Self {
        DataMut::TrueRef(data)
    }
}

impl<S> DataMut<'_, S> {
    #[inline]
    pub fn from_manually_drop(data: ManuallyDrop<S>) -> Self {
        DataMut::ManuallyDropOwned(data)
    }
}

impl<S> From<Arc<S>> for DataArc<S> {
    #[inline]
    fn from(data: Arc<S>) -> Self {
        Self { storage: data }
    }
}

impl<S> From<S> for DataArc<S> {
    #[inline]
    fn from(data: S) -> Self {
        Self { storage: Arc::new(data) }
    }
}

pub trait DataAPI {
    type Data: Clone;
    fn storage(&self) -> &Self::Data;
    fn into_owned(self) -> DataOwned<Self::Data>;
    fn into_shared(self) -> DataArc<Self::Data>;
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef::from(self.storage())
    }
}

pub trait DataMutAPI: DataAPI {
    fn storage_mut(&mut self) -> &mut Self::Data;
    fn as_mut(&mut self) -> DataMut<Self::Data> {
        DataMut::TrueRef(self.storage_mut())
    }
}

pub trait DataOwnedAPI: DataMutAPI {}

/* #region impl DataAPI */

impl<S> DataAPI for DataOwned<S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        &self.storage
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        DataArc::from(self.storage)
    }
}

impl<S> DataAPI for DataRef<'_, S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        match self {
            DataRef::TrueRef(storage) => storage,
            DataRef::ManuallyDropOwned(storage) => storage,
        }
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataRef::TrueRef(storage) => DataOwned::from(storage.clone()),
            DataRef::ManuallyDropOwned(storage) => {
                let v = ManuallyDrop::into_inner(storage);
                DataOwned::from(v.clone())
            },
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataRef::TrueRef(storage) => DataArc::from(storage.clone()),
            DataRef::ManuallyDropOwned(storage) => {
                let v = ManuallyDrop::into_inner(storage);
                DataArc::from(v.clone())
            },
        }
    }
}

impl<S> DataAPI for DataMut<'_, S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        match self {
            DataMut::TrueRef(storage) => storage,
            DataMut::ManuallyDropOwned(storage) => storage,
        }
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataMut::TrueRef(storage) => DataOwned::from(storage.clone()),
            DataMut::ManuallyDropOwned(storage) => {
                let v = ManuallyDrop::into_inner(storage);
                DataOwned::from(v.clone())
            },
        }
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        match self {
            DataMut::TrueRef(storage) => DataArc::from(storage.clone()),
            DataMut::ManuallyDropOwned(storage) => {
                let v = ManuallyDrop::into_inner(storage);
                DataArc::from(v.clone())
            },
        }
    }
}

impl<S> DataAPI for DataCow<'_, S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        match self {
            DataCow::Owned(data) => data.storage(),
            DataCow::Ref(data) => data.storage(),
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
            DataCow::Owned(data) => DataArc::from(data.into_storage()),
            DataCow::Ref(data) => data.into_shared(),
        }
    }
}

impl<S> DataAPI for DataArc<S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        &self.storage
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned::from(Arc::try_unwrap(self.storage).ok().unwrap())
    }

    #[inline]
    fn into_shared(self) -> DataArc<Self::Data> {
        self
    }
}

/* #endregion */

/* #region impl DataMutAPI */

impl<S> DataMutAPI for DataOwned<S>
where
    S: Clone,
{
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Data {
        &mut self.storage
    }
}

impl<S> DataMutAPI for DataMut<'_, S>
where
    S: Clone,
{
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Data {
        match self {
            DataMut::TrueRef(storage) => storage,
            DataMut::ManuallyDropOwned(storage) => storage,
        }
    }
}

impl<S> DataMutAPI for DataArc<S>
where
    S: Clone,
{
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Data {
        Arc::make_mut(&mut self.storage)
    }
}

/* #endregion */

/* #region DataCow */

impl<S> From<DataOwned<S>> for DataCow<'_, S> {
    #[inline]
    fn from(data: DataOwned<S>) -> Self {
        DataCow::Owned(data)
    }
}

impl<'a, S> From<DataRef<'a, S>> for DataCow<'a, S> {
    #[inline]
    fn from(data: DataRef<'a, S>) -> Self {
        DataCow::Ref(data)
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
        let data = DataOwned { storage: vec.clone() };
        let data_ref = data.as_ref();
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.storage().as_ptr());
        println!("{:?}", data_ref_ref.storage().as_ptr());
        let data_ref2 = data_ref.into_owned();
        println!("{:?}", data_ref2.storage().as_ptr());

        println!("===");
        let data_ref = DataRef::from_manually_drop(ManuallyDrop::new(vec.clone()));
        let data_ref_ref = data_ref.as_ref();
        println!("{:?}", data_ref.storage().as_ptr());
        println!("{:?}", data_ref_ref.storage().as_ptr());
        let mut data_ref2 = data_ref.into_owned();
        println!("{:?}", data_ref2.storage().as_ptr());
        data_ref2.storage_mut()[1] = 10;
    }
}
