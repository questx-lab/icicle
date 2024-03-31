use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostOrDeviceSlice2DMut};

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
pub struct MerkleTreeConfig<'a, F: FieldImpl> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    pub max_mimc_k: u32,

    // These are array addresses on device. Do not use them on host.
    pub mimc_params: *const F,
    pub d: *const u32,
}

impl<'a, F: FieldImpl> MerkleTreeConfig<'a, F> {
    pub fn default_for_device(mimc_params: &HostOrDeviceSlice<F>, d: &HostOrDeviceSlice<u32>) -> Self {
        Self {
            ctx: DeviceContext::default_for_device(DEFAULT_DEVICE_ID),
            max_mimc_k: mimc_params.len() as u32,
            mimc_params: mimc_params.as_ptr(),
            d: d.as_ptr(),
        }
    }
}

///////////// CIRCUIT

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SparseMultilinearExtension<F: FieldImpl> {
    pub size: u32,

    pub z_num_vars: u32,
    pub x_num_vars: u32,
    pub y_num_vars: u32,

    pub point_z: *const u32,
    pub point_x: *const u32,
    pub point_y: *const u32,
    pub evaluations: *const F,

    pub z_indices_size: *const u8,
    pub z_indices: *const *const u32,

    pub x_indices_size: *const u8,
    pub x_indices: *const *const u32,

    pub y_indices_size: *const u8,
    pub y_indices: *const *const u32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ReverseSparseMultilinearExtension {
    pub size: u32,

    pub subset_num_vars: u32,
    pub real_num_vars: u32,

    pub point_subset: *const u32,
    pub point_real: *const u32,

    pub subset_position: *const u32,
    pub real_position: *const u32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Layer<F: FieldImpl> {
    pub layer_index: u8,
    pub num_layers: u8,
    pub size: u32,

    pub constant_ext: *const SparseMultilinearExtension<F>,
    pub mul_ext: *const SparseMultilinearExtension<F>,
    pub forward_x_ext: *const SparseMultilinearExtension<F>,
    pub forward_y_ext: *const SparseMultilinearExtension<F>,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Circuit<F: FieldImpl> {
    pub num_layers: u8,
    pub layers: *const Layer<F>,
    pub on_host_subset_num_vars: *const *const u32,
    pub reverse_exts: *const *const ReverseSparseMultilinearExtension,
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

    fn bk_reduce(config: &SumcheckConfig, arr: &mut HostOrDeviceSlice<F>, n: u32, r: F) -> IcicleResult<()>;

    //// Merkle tree
    fn build_merkle_tree(config: &MerkleTreeConfig<F>, tree: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;

    fn hash_merkle_tree_slice(
        config: &MerkleTreeConfig<F>,
        input: &HostOrDeviceSlice<F>,
        output: &mut HostOrDeviceSlice<F>,
        n: u32,
        slice_size: u32,
    ) -> IcicleResult<()>;

    fn circuit_evaluate(circuit: &Circuit<F>, evaluations: &HostOrDeviceSlice2DMut<F>) -> IcicleResult<()>;
    fn circuit_subset_evaluations(
        circuit: &Circuit<F>,
        layer_index: u8,
        evaluations: &HostOrDeviceSlice2DMut<F>,
        subset_evaluations: &HostOrDeviceSlice2DMut<F>,
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

pub fn bk_reduce<F>(config: &SumcheckConfig, arr: &mut HostOrDeviceSlice<F>, n: u32, r: F) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::bk_reduce(config, arr, n, r)
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

pub fn build_merkle_tree<F>(config: &MerkleTreeConfig<F>, tree: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::build_merkle_tree(config, tree, n)
}

pub fn hash_merkle_tree_slice<F>(
    config: &MerkleTreeConfig<F>,
    input: &HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
    n: u32,
    slice_size: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    assert!(n % slice_size == 0);
    <<F as FieldImpl>::Config as Virgo<F>>::hash_merkle_tree_slice(config, input, output, n, slice_size)
}

pub fn circuit_evaluate<F>(circuit: &Circuit<F>, evaluations: &HostOrDeviceSlice2DMut<F>) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::circuit_evaluate(circuit, evaluations)
}

pub fn circuit_subset_evaluations<F>(
    circuit: &Circuit<F>,
    layer_index: u8,
    evaluations: &HostOrDeviceSlice2DMut<F>,
    subset_evaluations: &HostOrDeviceSlice2DMut<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::circuit_subset_evaluations(
        circuit,
        layer_index,
        evaluations,
        subset_evaluations,
    )
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
            use crate::virgo::{
                $field, $field_config, Circuit, CudaError, DeviceContext, MerkleTreeConfig, SumcheckConfig,
            };

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
                #[link_name = concat!($field_prefix, "BkReduce")]
                pub(crate) fn _bk_reduce(config: &SumcheckConfig, arr: *mut $field, n: u32, r: $field) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "BuildMerkleTree")]
                pub(crate) fn _build_merkle_tree(
                    config: &MerkleTreeConfig<$field>,
                    tree: *mut $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "HashMerkleTreeSlice")]
                pub(crate) fn _hash_merkle_tree_slice(
                    config: &MerkleTreeConfig<$field>,
                    input: *const $field,
                    output: *mut $field,
                    n: u32,
                    slice_size: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "CircuitEvaluate")]
                pub(crate) fn _circuit_evaluate(
                    circuit: &Circuit<$field>,
                    evaluations: *const *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "CircuitSubsetEvaluations")]
                pub(crate) fn _circuit_subset_evaluations(
                    circuit: &Circuit<$field>,
                    layer_index: u8,
                    evaluations: *const *mut $field,
                    subset_evaluations: *const *mut $field,
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

            fn bk_reduce(
                config: &SumcheckConfig,
                arr: &mut HostOrDeviceSlice<$field>,
                n: u32,
                r: $field,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_bk_reduce(config, arr.as_mut_ptr(), n, r).wrap() }
            }

            fn build_merkle_tree(
                config: &MerkleTreeConfig<$field>,
                tree: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_build_merkle_tree(config, tree.as_mut_ptr(), n).wrap() }
            }

            fn hash_merkle_tree_slice(
                config: &MerkleTreeConfig<$field>,
                input: &HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice<$field>,
                n: u32,
                slice_size: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_hash_merkle_tree_slice(
                        config,
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        n,
                        slice_size,
                    )
                    .wrap()
                }
            }

            fn circuit_evaluate(
                circuit: &Circuit<$field>,
                evaluations: &HostOrDeviceSlice2DMut<$field>,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_circuit_evaluate(circuit, evaluations.as_ptr()).wrap() }
            }

            fn circuit_subset_evaluations(
                circuit: &Circuit<$field>,
                layer_index: u8,
                evaluations: &HostOrDeviceSlice2DMut<$field>,
                subset_evaluations: &HostOrDeviceSlice2DMut<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_circuit_subset_evaluations(
                        circuit,
                        layer_index,
                        evaluations.as_ptr(),
                        subset_evaluations.as_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}
