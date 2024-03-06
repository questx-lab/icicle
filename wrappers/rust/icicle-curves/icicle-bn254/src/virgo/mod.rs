use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_virgo;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::virgo::Virgo;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_virgo!("bn254", bn254, ScalarField, ScalarCfg);
