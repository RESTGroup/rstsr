use crate::prelude_dev::*;

pub trait DeviceBaseAPI: Default + Clone {
    fn same_device(&self, other: &Self) -> bool;
}

pub trait DeviceRawAPI<T>: DeviceBaseAPI {
    type Raw: Clone;
}

#[derive(Debug)]
pub struct Storage<R, T, B>
where
    B: DeviceRawAPI<T>,
{
    pub(crate) data: R,
    pub(crate) device: B,
    _phantom: PhantomData<T>,
}

pub trait DeviceStorageAPI<T>: DeviceRawAPI<T> {
    fn len<R>(storage: &Storage<R, T, Self>) -> usize
    where
        R: DataAPI<Data = Self::Raw>;
    fn is_empty<R>(storage: &Storage<R, T, Self>) -> bool
    where
        R: DataAPI<Data = Self::Raw>,
    {
        Self::len::<R>(storage) == 0
    }
    fn to_cpu_vec<R>(storage: &Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>;
    fn into_cpu_vec<R>(storage: Storage<R, T, Self>) -> Result<Vec<T>>
    where
        R: DataAPI<Data = Self::Raw>;
    fn get_index<R>(storage: &Storage<R, T, Self>, index: usize) -> T
    where
        R: DataAPI<Data = Self::Raw>;
    fn get_index_ptr<R>(storage: &Storage<R, T, Self>, index: usize) -> *const T
    where
        R: DataAPI<Data = Self::Raw>;
    fn get_index_mut_ptr<R>(storage: &mut Storage<R, T, Self>, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = Self::Raw>;
    fn set_index<R>(storage: &mut Storage<R, T, Self>, index: usize, value: T)
    where
        R: DataMutAPI<Data = Self::Raw>;
}

impl<R, T, B> Storage<R, T, B>
where
    B: DeviceStorageAPI<T>,
    R: DataAPI<Data = B::Raw>,
{
    pub fn device(&self) -> &B {
        &self.device
    }

    pub fn data(&self) -> &R {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut R {
        &mut self.data
    }

    pub fn into_raw_parts(self) -> (R, B) {
        (self.data, self.device)
    }

    pub fn new(data: R, device: B) -> Self {
        Self { data, device, _phantom: PhantomData }
    }

    pub fn len(&self) -> usize {
        B::len(self)
    }

    pub fn is_empty(&self) -> bool {
        B::is_empty(self)
    }

    pub fn to_cpu_vec(&self) -> Result<Vec<T>> {
        B::to_cpu_vec(self)
    }

    pub fn into_cpu_vec(self) -> Result<Vec<T>> {
        B::into_cpu_vec(self)
    }

    #[inline]
    pub fn get_index(&self, index: usize) -> T {
        B::get_index(self, index)
    }

    #[inline]
    pub fn get_index_ptr(&self, index: usize) -> *const T {
        B::get_index_ptr(self, index)
    }

    #[inline]
    pub fn get_index_mut_ptr(&mut self, index: usize) -> *mut T
    where
        R: DataMutAPI<Data = B::Raw>,
    {
        B::get_index_mut_ptr(self, index)
    }

    #[inline]
    pub fn set_index(&mut self, index: usize, value: T)
    where
        R: DataMutAPI<Data = B::Raw>,
    {
        B::set_index(self, index, value)
    }
}

impl<R, T, B> Storage<R, T, B>
where
    R: DataAPI<Data = B::Raw>,
    B: DeviceRawAPI<T>,
{
    pub fn raw(&self) -> &B::Raw {
        self.data.raw()
    }
}

impl<R, T, B> Storage<R, T, B>
where
    R: DataMutAPI<Data = B::Raw>,
    B: DeviceRawAPI<T>,
{
    pub fn raw_mut(&mut self) -> &mut B::Raw {
        self.data.raw_mut()
    }
}

pub trait DeviceAPI<T>: DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> {}

/// Unique identifier for cuda devices.
///
/// This code is from `candle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    #[allow(unused)]
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use core::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// Conversion API for device storage.
pub trait DeviceStorageConversionAPI<R, T, B>
where
    Self: DeviceRawAPI<T>,
{
    fn into_device(storage: Storage<R, T, Self>, device: &B) -> Result<Storage<R, T, B>>
    where
        B: DeviceRawAPI<T>;
}
