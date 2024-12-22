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
pub enum DataRefMut<'a, S> {
    TrueRef(&'a mut S),
    ManuallyDropOwned(ManuallyDrop<S>),
}

#[derive(Debug)]
pub enum DataMutable<'a, S> {
    Owned(DataOwned<S>),
    RefMut(DataRefMut<'a, S>),
    ToBeCloned(DataRef<'a, S>, DataOwned<S>),
}

#[derive(Debug, Clone)]
pub enum DataCow<'a, S> {
    Owned(DataOwned<S>),
    Ref(DataRef<'a, S>),
}

#[derive(Debug)]
pub enum DataReference<'a, S> {
    Ref(DataRef<'a, S>),
    RefMut(DataRefMut<'a, S>),
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

impl<S> DataRefMut<'_, S> {
    #[inline]
    pub fn from_manually_drop(data: ManuallyDrop<S>) -> Self {
        DataRefMut::ManuallyDropOwned(data)
    }
}

pub trait DataAPI {
    type Data: Clone;
    fn storage(&self) -> &Self::Data;
    fn as_ref(&self) -> DataRef<Self::Data>;
    fn into_owned(self) -> DataOwned<Self::Data>;
}

pub trait DataMutAPI: DataAPI {
    fn storage_mut(&mut self) -> &mut Self::Data;
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data>;
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
    fn as_ref(&self) -> DataRef<Self::Data> {
        DataRef::from(&self.storage)
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        self
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
    fn as_ref(&self) -> DataRef<Self::Data> {
        match self {
            DataRef::TrueRef(storage) => DataRef::TrueRef(storage),
            DataRef::ManuallyDropOwned(storage) => DataRef::TrueRef(storage),
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
}

impl<S> DataAPI for DataRefMut<'_, S>
where
    S: Clone,
{
    type Data = S;

    #[inline]
    fn storage(&self) -> &Self::Data {
        match self {
            DataRefMut::TrueRef(storage) => storage,
            DataRefMut::ManuallyDropOwned(storage) => storage,
        }
    }

    #[inline]
    fn as_ref(&self) -> DataRef<'_, Self::Data> {
        DataRef::from(self.storage())
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        DataOwned::from(self.storage().clone())
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
    fn as_ref(&self) -> DataRef<Self::Data> {
        match self {
            DataCow::Owned(data) => data.as_ref(),
            DataCow::Ref(data) => data.as_ref(),
        }
    }

    #[inline]
    fn into_owned(self) -> DataOwned<Self::Data> {
        match self {
            DataCow::Owned(data) => data,
            DataCow::Ref(data) => data.into_owned(),
        }
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

    #[inline]
    fn as_ref_mut(&mut self) -> DataRefMut<Self::Data> {
        DataRefMut::TrueRef(&mut self.storage)
    }
}

impl<S> DataMutAPI for DataRefMut<'_, S>
where
    S: Clone,
{
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Data {
        match self {
            DataRefMut::TrueRef(storage) => storage,
            DataRefMut::ManuallyDropOwned(storage) => storage,
        }
    }

    #[inline]
    fn as_ref_mut(&mut self) -> DataRefMut<'_, Self::Data> {
        match self {
            DataRefMut::TrueRef(storage) => DataRefMut::TrueRef(storage),
            DataRefMut::ManuallyDropOwned(storage) => DataRefMut::TrueRef(storage),
        }
    }
}

/* #endregion */

/* #region DataOwnedAPI */

impl<S> DataOwnedAPI for DataOwned<S> where S: Clone {}

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
