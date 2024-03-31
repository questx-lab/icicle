use crate::bindings::{
    cudaError, cudaFree, cudaMalloc, cudaMallocAsync, cudaMemPool_t, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind,
};
use crate::device::get_device;
use crate::device_context::check_device;
use crate::error::{CudaError, CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use std::mem::{size_of, MaybeUninit};
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::os::raw::c_void;

pub trait ToCuda {
    type CudaRepr;

    fn to_cuda(&self) -> Self::CudaRepr;
}

pub struct HostOrDeviceSliceWrapper<W: ToCuda> {
    // Hold the origin device slices to not drop them.
    origin: Vec<W>,
    ptr: HostOrDeviceSlice<W::CudaRepr>,
}

impl<W: ToCuda> ToCuda for HostOrDeviceSliceWrapper<W> {
    type CudaRepr = *const W::CudaRepr;

    fn to_cuda(&self) -> Self::CudaRepr {
        self.as_ptr()
    }
}

impl<W: ToCuda> HostOrDeviceSliceWrapper<W> {
    pub fn new(val: Vec<W>) -> Self {
        let mut origin_ptr = vec![];
        for i in 0..val.len() {
            origin_ptr.push(val[i].to_cuda());
        }

        let ptr = HostOrDeviceSlice::on_device(&origin_ptr);

        Self { origin: val, ptr }
    }

    pub fn as_ptr(&self) -> *const W::CudaRepr {
        self.ptr
            .as_ptr()
    }
}

impl<'a, W: ToCuda> Index<usize> for HostOrDeviceSliceWrapper<W> {
    type Output = W;

    fn index(&self, index: usize) -> &Self::Output {
        &self.origin[index]
    }
}

pub struct HostOrDeviceSlice2DConst<T> {
    // Hold the origin device slices to not drop them.
    origin: Vec<HostOrDeviceSlice<T>>,
    ptr: HostOrDeviceSlice<*const T>,
}

impl<T> HostOrDeviceSlice2DConst<T> {
    pub fn new(val: Vec<Vec<T>>) -> Self {
        let mut origin = vec![];
        let mut origin_ptr = vec![];
        for i in 0..val.len() {
            let device = HostOrDeviceSlice::on_device(&val[i]);

            origin_ptr.push(device.as_ptr());
            origin.push(device);
        }

        let ptr = HostOrDeviceSlice::on_device(&origin_ptr);

        Self { origin, ptr }
    }

    pub fn as_ptr(&self) -> *const *const T {
        self.ptr
            .as_ptr()
    }

    pub fn len(&self) -> usize {
        self.origin
            .len()
    }

    pub fn into_iter(self) -> std::vec::IntoIter<HostOrDeviceSlice<T>> {
        self.origin
            .into_iter()
    }
}

impl<T> Index<usize> for HostOrDeviceSlice2DConst<T> {
    type Output = HostOrDeviceSlice<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.origin[index]
    }
}

pub struct HostOrDeviceSlice2DMut<T> {
    // Hold the origin device slices to not drop them.
    origin: Vec<HostOrDeviceSlice<T>>,
    ptr: HostOrDeviceSlice<*mut T>,
}

impl<T> HostOrDeviceSlice2DMut<T> {
    pub fn new(val: Vec<Vec<T>>) -> Self {
        let mut origin = vec![];
        let mut origin_ptr = vec![];
        for i in 0..val.len() {
            let mut device = HostOrDeviceSlice::on_device(&val[i]);

            origin_ptr.push(device.as_mut_ptr());
            origin.push(device);
        }

        let ptr = HostOrDeviceSlice::on_device(&origin_ptr);

        Self { origin, ptr }
    }

    pub fn as_ptr(&self) -> *const *mut T {
        self.ptr
            .as_ptr()
    }

    pub fn len(&self) -> usize {
        self.origin
            .len()
    }

    pub fn into_iter(self) -> std::vec::IntoIter<HostOrDeviceSlice<T>> {
        self.origin
            .into_iter()
    }
}

impl<T> Index<usize> for HostOrDeviceSlice2DMut<T> {
    type Output = HostOrDeviceSlice<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.origin[index]
    }
}

pub enum HostOrDeviceSlice<T> {
    Host(Vec<T>),
    Device(Option<(*mut T, usize)>, i32),
}

impl<T> HostOrDeviceSlice<T> {
    // Function to get the device_id for Device variant
    pub fn get_device_id(&self) -> Option<i32> {
        match self {
            HostOrDeviceSlice::Device(_, device_id) => Some(*device_id),
            HostOrDeviceSlice::Host(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Device(s, _) => {
                if s.is_none() {
                    0
                } else {
                    s.as_ref()
                        .unwrap()
                        .1
                }
            }
            Self::Host(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Device(s, _) => {
                if s.is_none() {
                    true
                } else {
                    s.as_ref()
                        .unwrap()
                        .1
                        == 0
                }
            }
            Self::Host(v) => v.is_empty(),
        }
    }

    pub fn is_on_device(&self) -> bool {
        match self {
            Self::Device(_, _) => true,
            Self::Host(_) => false,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Device(_, _) => panic!("Use copy_to_host and copy_to_host_async to move device data to a slice"),
            Self::Host(v) => v.as_mut_slice(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Device(_, _) => panic!("Use copy_to_host and copy_to_host_async to move device data to a slice"),
            Self::Host(v) => v.as_slice(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        if self.len() == 0 {
            return ::std::ptr::null();
        }

        match self {
            Self::Device(s, _) => {
                s.as_ref()
                    .unwrap()
                    .0
            }
            Self::Host(v) => v.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.len() == 0 {
            return ::std::ptr::null_mut();
        }

        match self {
            Self::Device(s, _) => {
                s.as_mut()
                    .unwrap()
                    .0
            }
            Self::Host(v) => v.as_mut_ptr(),
        }
    }

    pub fn on_device<'b>(v: &'b [T]) -> HostOrDeviceSlice<T> {
        let mut device = HostOrDeviceSlice::cuda_malloc(v.len()).unwrap();
        if v.len() > 0 {
            device
                .copy_from_host(v)
                .unwrap();
        }

        device
    }

    pub fn on_host(src: Vec<T>) -> Self {
        //TODO: HostOrDeviceSlice on_host() with slice input without actually copying the data
        Self::Host(src)
    }

    pub fn cuda_malloc(count: usize) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);

        if size == 0 {
            return Ok(Self::Device(None, get_device().unwrap() as i32));
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMalloc(device_ptr.as_mut_ptr(), size).wrap()?;
            Ok(Self::Device(
                Some((device_ptr.assume_init() as *mut T, count)),
                get_device().unwrap() as i32,
            ))
        }
    }

    pub fn cuda_malloc_async(count: usize, stream: &CudaStream) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation);
        }

        if size == 0 {
            return Ok(Self::Device(None, get_device().unwrap() as i32));
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMallocAsync(device_ptr.as_mut_ptr(), size, stream.handle as *mut _ as *mut _).wrap()?;
            Ok(Self::Device(
                Some((device_ptr.assume_init() as *mut T, count)),
                get_device().unwrap() as i32,
            ))
        }
    }

    pub fn copy_from_host(&mut self, val: &[T]) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_from_host_partially(&mut self, val: &[T]) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        assert!(self.len() >= val.len(), "Destination has a larger size than source");

        let size = size_of::<T>() * val.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host(&self, val: &mut [T]) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host_at_index(&self, val: &mut [T], val_index: usize, device_index: usize) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };

        let size = size_of::<T>() * val.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.as_mut_ptr()
                        .wrapping_add(val_index) as *mut c_void,
                    self.as_ptr()
                        .wrapping_add(device_index) as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_from_host_async(&mut self, val: &[T], stream: &CudaStream) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host_async(&self, val: &mut [T], stream: &CudaStream) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }
}

impl<T: Clone> Clone for HostOrDeviceSlice<T> {
    fn clone(&self) -> Self {
        match self {
            HostOrDeviceSlice::Host(v) => Self::Host(v.to_vec()),
            HostOrDeviceSlice::Device(s, device_id) => match s {
                None => Self::Device(None, device_id.clone()),
                Some((pointer, size)) => {
                    let mut new = HostOrDeviceSlice::cuda_malloc(*size).unwrap();

                    let count = size_of::<T>() * self.len();
                    unsafe {
                        let err = cudaMemcpy(
                            new.as_mut_ptr() as *mut c_void,
                            *pointer as *const c_void,
                            count,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        );

                        if err != cudaError::cudaSuccess {
                            panic!("failed to clone device memory {:?}", err);
                        }
                    }

                    new
                }
            },
        }
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T> Index<$t> for HostOrDeviceSlice<T>
            {
                type Output = [T];

                fn index(&self, index: $t) -> &Self::Output {
                    match self {
                        Self::Device(_, _) => panic!("doesn't support index device memory"),
                        Self::Host(v) => v.index(index),
                    }
                }
            }

            impl<T> IndexMut<$t> for HostOrDeviceSlice<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self::Output {
                    match self {
                        Self::Device(_,_) => panic!("doesn't support index device memory"),
                        Self::Host(v) => v.index_mut(index),
                    }
                }
            }
        )*
    }
}

unsafe impl<T> Send for HostOrDeviceSlice<T> {}
unsafe impl<T> Sync for HostOrDeviceSlice<T> {}

impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<T> Drop for HostOrDeviceSlice<T> {
    fn drop(&mut self) {
        match self {
            Self::Device(option_s, device_id) => match option_s {
                Some((s, _)) => {
                    check_device(*device_id);

                    unsafe {
                        cudaFree(*s as *mut c_void)
                            .wrap()
                            .unwrap();
                    }
                }
                _ => {}
            },
            Self::Host(_) => {}
        }
    }
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = cudaMemPool_t;
