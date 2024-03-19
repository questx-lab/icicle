use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{error::IcicleResult, traits::FieldImpl};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SumcheckConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
}

impl<'a> Default for SumcheckConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> SumcheckConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(device_id),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MerkleTreeConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    pub MAX_MIMC_K: u32,
}

impl<'a> Default for MerkleTreeConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> MerkleTreeConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(device_id),
            MAX_MIMC_K: 128,
        }
    }
}

/////////////

pub trait Virgo<F: FieldImpl> {
    fn bk_sum_all_case_1(
        config: &SumcheckConfig,
        table1: &HostOrDeviceSlice<F>,
        table2: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;

    fn bk_sum_all_case_2(
        config: &SumcheckConfig,
        table: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;

    fn bk_produce_case_1(
        config: &SumcheckConfig,
        table1: &HostOrDeviceSlice<F>,
        table2: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;

    fn bk_produce_case_2(
        config: &SumcheckConfig,
        table: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;

    //// Merkle tree
    fn build_merkle_tree(
        config: &MerkleTreeConfig,
        table: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;
}

pub fn bk_sum_all_case_1<F>(
    config: &SumcheckConfig,
    table1: &HostOrDeviceSlice<F>,
    table2: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::bk_sum_all_case_1(config, table1, table2, result, n)
}

pub fn bk_sum_all_case_2<F>(
    config: &SumcheckConfig,
    table: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::bk_sum_all_case_2(config, table, result, n)
}

pub fn bk_produce_case_1<F>(
    config: &SumcheckConfig,
    table1: &HostOrDeviceSlice<F>,
    table2: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::bk_produce_case_1(config, table1, table2, result, n)
}

pub fn bk_produce_case_2<F>(
    config: &SumcheckConfig,
    table: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::bk_produce_case_2(config, table, result, n)
}

pub fn build_merkle_tree<F>(
    config: &MerkleTreeConfig,
    table: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::build_merkle_tree(config, table, result, n)
}

#[macro_export]
macro_rules! impl_virgo {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
      ) => {
        mod $field_prefix_ident {
            use crate::virgo::{$field, $field_config, CudaError, DeviceContext, MerkleTreeConfig, SumcheckConfig};

            extern "C" {
                #[link_name = concat!($field_prefix, "BkSumAllCase1")]
                pub(crate) fn _bk_sum_all_case_1(
                    config: &SumcheckConfig,
                    table1: *const $field,
                    table2: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "BkSumAllCase2")]
                pub(crate) fn _bk_sum_all_case_2(
                    config: &SumcheckConfig,
                    arr: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "BkProduceCase1")]
                pub(crate) fn _bk_produce_case_1(
                    config: &SumcheckConfig,
                    table1: *const $field,
                    table2: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "BkProduceCase2")]
                pub(crate) fn _bk_produce_case_2(
                    config: &SumcheckConfig,
                    arr: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "BuildMerkleTree")]
                pub(crate) fn _build_merkle_tree(
                    config: &MerkleTreeConfig,
                    arr: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }
        }

        impl Virgo<$field> for $field_config {
            fn bk_sum_all_case_1(
                config: &SumcheckConfig,
                table1: &HostOrDeviceSlice<$field>,
                table2: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_bk_sum_all_case_1(
                        config,
                        table1.as_ptr(),
                        table2.as_ptr(),
                        result.as_mut_ptr(),
                        n,
                    )
                    .wrap()
                }
            }

            fn bk_sum_all_case_2(
                config: &SumcheckConfig,
                table: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_bk_sum_all_case_2(config, table.as_ptr(), result.as_mut_ptr(), n).wrap()
                }
            }

            fn bk_produce_case_1(
                config: &SumcheckConfig,
                table1: &HostOrDeviceSlice<$field>,
                table2: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_bk_produce_case_1(
                        config,
                        table1.as_ptr(),
                        table2.as_ptr(),
                        result.as_mut_ptr(),
                        n,
                    )
                    .wrap()
                }
            }

            fn bk_produce_case_2(
                config: &SumcheckConfig,
                table: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_bk_produce_case_2(config, table.as_ptr(), result.as_mut_ptr(), n).wrap()
                }
            }

            fn build_merkle_tree(
                config: &MerkleTreeConfig,
                table: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_build_merkle_tree(config, table.as_ptr(), result.as_mut_ptr(), n).wrap()
                }
            }
        }
    };
}
