use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::{
    error::IcicleResult,
    impl_sisu,
    sisu::{
        Circuit, MerkleTreeConfig, ReverseSparseMultilinearExtension, Sisu, SparseMultilinearExtension, SumcheckConfig,
    },
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::{
    device_context::DeviceContext,
    error::CudaError,
    memory::{HostOrDeviceSlice, HostOrDeviceSlice2D},
};

impl_sisu!("bn254", bn254, ScalarField, ScalarCfg);
