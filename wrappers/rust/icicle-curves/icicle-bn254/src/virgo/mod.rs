use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::{
    error::IcicleResult,
    impl_virgo,
    traits::IcicleResultWrap,
    virgo::{Circuit, MerkleTreeConfig, SparseMultilinearExtension, SumcheckConfig, Virgo},
};
use icicle_cuda_runtime::{
    device_context::DeviceContext,
    error::CudaError,
    memory::{HostOrDeviceSlice, HostOrDeviceSlice2D},
};

impl_virgo!("bn254", bn254, ScalarField, ScalarCfg);
