use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostOrDeviceSlice2D};

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

    pub z_indices_start: *const u32,
    pub z_indices: *const u32,

    pub x_indices_start: *const u32,
    pub x_indices: *const u32,

    pub y_indices_start: *const u32,
    pub y_indices: *const u32,
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

pub trait Sisu<F: FieldImpl> {
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

    fn exchange_evaluations(
        evaluations: &HostOrDeviceSlice2D<F>,
        output: &mut HostOrDeviceSlice<F>,
    ) -> IcicleResult<()>;

    fn circuit_evaluate(
        circuit: &Circuit<F>,
        num_circuits: usize,
        evaluations: &mut HostOrDeviceSlice2D<F>,
    ) -> IcicleResult<()>;

    fn circuit_subset_evaluations(
        circuit: &Circuit<F>,
        num_circuits: usize,
        layer_index: u8,
        evaluations: &HostOrDeviceSlice2D<F>,
        subset_evaluations: &mut HostOrDeviceSlice2D<F>,
    ) -> IcicleResult<()>;

    fn mul_by_scalar(arr: &mut HostOrDeviceSlice<F>, scalar: F, n: u32) -> IcicleResult<()>;

    fn precompute_bookeeping(
        init: F,
        g: &HostOrDeviceSlice<F>,
        g_size: u8,
        output: &mut HostOrDeviceSlice<F>,
    ) -> IcicleResult<()>;

    fn initialize_phase_1_plus(
        num_layers: u32,
        output_size: u32,
        f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<F>>,
        s_evaluations: &HostOrDeviceSlice2D<F>,
        bookeeping_g: &HostOrDeviceSlice<F>,
        output: &mut HostOrDeviceSlice<F>,
    ) -> IcicleResult<()>;

    fn initialize_phase_2_plus(
        num_layers: u32,
        on_host_output_size: Vec<u32>,
        f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<F>>,
        bookeeping_g: &HostOrDeviceSlice<F>,
        bookeeping_u: &HostOrDeviceSlice<F>,
        output: &mut HostOrDeviceSlice2D<F>,
    ) -> IcicleResult<()>;

    fn initialize_combining_point(
        num_layers: u32,
        on_host_bookeeping_rs_size: Vec<u32>,
        bookeeping_rs: &HostOrDeviceSlice2D<F>,
        reverse_exts: &HostOrDeviceSlice<ReverseSparseMultilinearExtension>,
        output: &mut HostOrDeviceSlice<F>,
    ) -> IcicleResult<()>;

    fn fold_multi(
        domain: &HostOrDeviceSlice<F>,
        random_point: F,
        evaluations: &HostOrDeviceSlice2D<F>,
        output: &mut HostOrDeviceSlice2D<F>,
    ) -> IcicleResult<()>;

    fn dense_mle_multi(
        output: &mut HostOrDeviceSlice<F>,
        evaluations: HostOrDeviceSlice2D<F>,
        input: Vec<F>,
    ) -> IcicleResult<()>;

    fn mul_arr_multi(a: &mut HostOrDeviceSlice2D<F>, b: &HostOrDeviceSlice2D<F>) -> IcicleResult<()>;

    fn sub_arr_multi(a: &mut HostOrDeviceSlice2D<F>, b: &HostOrDeviceSlice2D<F>) -> IcicleResult<()>;

    fn mul_by_scalar_multi(arr: &mut HostOrDeviceSlice2D<F>, scalar: F) -> IcicleResult<()>;
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
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::bk_sum_all_case_1(config, table1, table2, result, n)
}

pub fn bk_sum_all_case_2<F>(
    config: &SumcheckConfig,
    table: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::bk_sum_all_case_2(config, table, result, n)
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
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::bk_produce_case_1(config, table1, table2, result, n)
}

pub fn bk_reduce<F>(config: &SumcheckConfig, arr: &mut HostOrDeviceSlice<F>, n: u32, r: F) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::bk_reduce(config, arr, n, r)
}

pub fn bk_produce_case_2<F>(
    config: &SumcheckConfig,
    table: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::bk_produce_case_2(config, table, result, n)
}

pub fn build_merkle_tree<F>(config: &MerkleTreeConfig<F>, tree: &mut HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::build_merkle_tree(config, tree, n)
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
    <F as FieldImpl>::Config: Sisu<F>,
{
    assert!(n % slice_size == 0);
    <<F as FieldImpl>::Config as Sisu<F>>::hash_merkle_tree_slice(config, input, output, n, slice_size)
}

pub fn circuit_evaluate<F>(
    circuit: &Circuit<F>,
    num_circuits: usize,
    evaluations: &mut HostOrDeviceSlice2D<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::circuit_evaluate(circuit, num_circuits, evaluations)
}

pub fn circuit_subset_evaluations<F>(
    circuit: &Circuit<F>,
    num_circuits: usize,
    layer_index: u8,
    evaluations: &HostOrDeviceSlice2D<F>,
    subset_evaluations: &mut HostOrDeviceSlice2D<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::circuit_subset_evaluations(
        circuit,
        num_circuits,
        layer_index,
        evaluations,
        subset_evaluations,
    )
}

pub fn mul_by_scalar<F>(arr: &mut HostOrDeviceSlice<F>, scalar: F) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::mul_by_scalar(arr, scalar, arr.len() as u32)
}

pub fn precompute_bookeeping<F>(
    init: F,
    g: &HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::precompute_bookeeping(init, g, g.len() as u8, output)
}

pub fn initialize_phase_1_plus<F>(
    num_layers: u32,
    output_size: u32,
    f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<F>>,
    s_evaluations: &HostOrDeviceSlice2D<F>,
    bookeeping_g: &HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::initialize_phase_1_plus(
        num_layers,
        output_size,
        f_extensions,
        s_evaluations,
        bookeeping_g,
        output,
    )
}

pub fn initialize_phase_2_plus<F>(
    num_layers: u32,
    on_host_output_size: Vec<u32>,
    f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<F>>,
    bookeeping_g: &HostOrDeviceSlice<F>,
    bookeeping_u: &HostOrDeviceSlice<F>,
    output: &mut HostOrDeviceSlice2D<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::initialize_phase_2_plus(
        num_layers,
        on_host_output_size,
        f_extensions,
        bookeeping_g,
        bookeeping_u,
        output,
    )
}

pub fn initialize_combining_point<F>(
    num_layers: u32,
    on_host_bookeeping_rs_size: Vec<u32>,
    bookeeping_rs: &HostOrDeviceSlice2D<F>,
    reverse_exts: &HostOrDeviceSlice<ReverseSparseMultilinearExtension>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::initialize_combining_point(
        num_layers,
        on_host_bookeeping_rs_size,
        bookeeping_rs,
        reverse_exts,
        output,
    )
}

pub fn exchange_evaluations<F>(
    evaluations: &HostOrDeviceSlice2D<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::exchange_evaluations(evaluations, output)
}

pub fn fold_multi<F>(
    domain: &HostOrDeviceSlice<F>,
    random_point: F,
    evaluations: &HostOrDeviceSlice2D<F>,
    output: &mut HostOrDeviceSlice2D<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::fold_multi(domain, random_point, evaluations, output)
}

pub fn dense_mle_multi<F>(
    output: &mut HostOrDeviceSlice<F>,
    evaluations: HostOrDeviceSlice2D<F>,
    input: Vec<F>,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::dense_mle_multi(output, evaluations, input)
}

pub fn mul_arr_multi<F>(a: &mut HostOrDeviceSlice2D<F>, b: &HostOrDeviceSlice2D<F>) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::mul_arr_multi(a, b)
}

pub fn sub_arr_multi<F>(a: &mut HostOrDeviceSlice2D<F>, b: &HostOrDeviceSlice2D<F>) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::sub_arr_multi(a, b)
}

pub fn mul_by_scalar_multi<F>(a: &mut HostOrDeviceSlice2D<F>, scalar: F) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Sisu<F>,
{
    <<F as FieldImpl>::Config as Sisu<F>>::mul_by_scalar_multi(a, scalar)
}

#[macro_export]
macro_rules! impl_sisu {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
      ) => {
        mod $field_prefix_ident {
            use crate::sisu::{
                $field, $field_config, Circuit, CudaError, DeviceContext, MerkleTreeConfig,
                ReverseSparseMultilinearExtension, SparseMultilinearExtension, SumcheckConfig,
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
                #[link_name = concat!($field_prefix, "ExchangeEvaluations")]
                pub(crate) fn _exchange_evaluations(
                    num_evaluations: u32,
                    evaluation_size: u32,
                    evaluations: *const *const $field,
                    output: *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "CircuitEvaluate")]
                pub(crate) fn _circuit_evaluate(
                    circuit: &Circuit<$field>,
                    num_subcircuits: u32,
                    evaluations: *const *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "CircuitSubsetEvaluations")]
                pub(crate) fn _circuit_subset_evaluations(
                    circuit: &Circuit<$field>,
                    num_subcircuits: u32,
                    layer_index: u8,
                    evaluations: *const *const $field,
                    subset_evaluations: *const *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "MulByScalar")]
                pub(crate) fn _mul_by_scalar(evaluations: *mut $field, scalar: $field, n: u32) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "PrecomputeBookeeping")]
                pub(crate) fn _precompute_bookeeping(
                    init: $field,
                    g: *const $field,
                    g_size: u8,
                    output: *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "InitializePhase1Plus")]
                pub(crate) fn _initialize_phase_1_plus(
                    num_layers: u32,
                    output_size: u32,
                    f_extensions: *const SparseMultilinearExtension<$field>,
                    s_evaluations: *const *const $field,
                    bookeeping_g: *const $field,
                    output: *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "InitializePhase2Plus")]
                pub(crate) fn _initialize_phase_2_plus(
                    num_layers: u32,
                    output_size: *const u32,
                    f_extensions: *const SparseMultilinearExtension<$field>,
                    bookeeping_g: *const $field,
                    bookeeping_u: *const $field,
                    output: *const *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "InitializeCombiningPoint")]
                pub(crate) fn _initialize_combining_point(
                    num_layers: u32,
                    on_host_bookeeping_rs_size: *const u32,
                    bookeeping_rs: *const *const $field,
                    reverse_exts: *const ReverseSparseMultilinearExtension,
                    output: *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FoldMulti")]
                pub(crate) fn _fold_multi(
                    domain: *const $field,
                    domain_size: u32,
                    num_replicas: u32,
                    random_point: $field,
                    evaluations: *const *const $field,
                    evaluation_size: u32,
                    output: *const *mut $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "DenseMleMulti")]
                pub(crate) fn _dense_mle_multi(
                    num_mle: u32,
                    output: *mut $field,
                    evaluations: *const *mut $field,
                    evaluation_size: u32,
                    on_host_input: *const $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "MulArrMulti")]
                pub(crate) fn _mul_arr_multi(
                    num_arr: u32,
                    size: u32,
                    a: *const *mut $field,
                    b: *const *const $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "SubArrMulti")]
                pub(crate) fn _sub_arr_multi(
                    num_arr: u32,
                    size: u32,
                    a: *const *mut $field,
                    b: *const *const $field,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "MulByScalarMulti")]
                pub(crate) fn _mul_by_scalar_multi(
                    num_arr: u32,
                    size: u32,
                    a: *const *mut $field,
                    scalar: $field,
                ) -> CudaError;
            }
        }

        impl Sisu<$field> for $field_config {
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

            fn exchange_evaluations(
                evaluations: &HostOrDeviceSlice2D<$field>,
                output: &mut HostOrDeviceSlice<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_exchange_evaluations(
                        evaluations.len() as u32,
                        evaluations[0].len() as u32,
                        evaluations.as_ptr(),
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn circuit_evaluate(
                circuit: &Circuit<$field>,
                num_circuits: usize,
                evaluations: &mut HostOrDeviceSlice2D<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_circuit_evaluate(circuit, num_circuits as u32, evaluations.as_mut_ptr())
                        .wrap()
                }
            }

            fn circuit_subset_evaluations(
                circuit: &Circuit<$field>,
                num_circuits: usize,
                layer_index: u8,
                evaluations: &HostOrDeviceSlice2D<$field>,
                subset_evaluations: &mut HostOrDeviceSlice2D<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_circuit_subset_evaluations(
                        circuit,
                        num_circuits as u32,
                        layer_index,
                        evaluations.as_ptr(),
                        subset_evaluations.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul_by_scalar(arr: &mut HostOrDeviceSlice<$field>, scalar: $field, n: u32) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_mul_by_scalar(arr.as_mut_ptr(), scalar, n).wrap() }
            }

            fn precompute_bookeeping(
                init: $field,
                g: &HostOrDeviceSlice<$field>,
                g_size: u8,
                output: &mut HostOrDeviceSlice<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_precompute_bookeeping(init, g.as_ptr(), g_size, output.as_mut_ptr()).wrap()
                }
            }

            fn initialize_phase_1_plus(
                num_layers: u32,
                output_size: u32,
                f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<$field>>,
                s_evaluations: &HostOrDeviceSlice2D<$field>,
                bookeeping_g: &HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_initialize_phase_1_plus(
                        num_layers,
                        output_size,
                        f_extensions.as_ptr(),
                        s_evaluations.as_ptr(),
                        bookeeping_g.as_ptr(),
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn initialize_phase_2_plus(
                num_layers: u32,
                on_host_output_size: Vec<u32>,
                f_extensions: &HostOrDeviceSlice<SparseMultilinearExtension<$field>>,
                bookeeping_g: &HostOrDeviceSlice<$field>,
                bookeeping_u: &HostOrDeviceSlice<$field>,
                output: &mut HostOrDeviceSlice2D<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_initialize_phase_2_plus(
                        num_layers,
                        on_host_output_size.as_ptr(),
                        f_extensions.as_ptr(),
                        bookeeping_g.as_ptr(),
                        bookeeping_u.as_ptr(),
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn initialize_combining_point(
                num_layers: u32,
                on_host_bookeeping_rs_size: Vec<u32>,
                bookeeping_rs: &HostOrDeviceSlice2D<$field>,
                reverse_exts: &HostOrDeviceSlice<ReverseSparseMultilinearExtension>,
                output: &mut HostOrDeviceSlice<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_initialize_combining_point(
                        num_layers,
                        on_host_bookeeping_rs_size.as_ptr(),
                        bookeeping_rs.as_ptr(),
                        reverse_exts.as_ptr(),
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn fold_multi(
                domain: &HostOrDeviceSlice<$field>,
                random_point: $field,
                evaluations: &HostOrDeviceSlice2D<$field>,
                output: &mut HostOrDeviceSlice2D<$field>,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_fold_multi(
                        domain.as_ptr(),
                        domain.len() as u32,
                        evaluations.len() as u32,
                        random_point,
                        evaluations.as_ptr(),
                        evaluations[0].len() as u32,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn dense_mle_multi(
                output: &mut HostOrDeviceSlice<$field>,
                mut evaluations: HostOrDeviceSlice2D<$field>,
                on_host_input: Vec<$field>,
            ) -> IcicleResult<()> {
                let evaluation_size = evaluations[0].len();

                unsafe {
                    $field_prefix_ident::_dense_mle_multi(
                        output.len() as u32,
                        output.as_mut_ptr(),
                        evaluations.as_mut_ptr(),
                        evaluation_size as u32,
                        on_host_input.as_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul_arr_multi(a: &mut HostOrDeviceSlice2D<$field>, b: &HostOrDeviceSlice2D<$field>) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_mul_arr_multi(a.len() as u32, a[0].len() as u32, a.as_mut_ptr(), b.as_ptr())
                        .wrap()
                }
            }

            fn sub_arr_multi(a: &mut HostOrDeviceSlice2D<$field>, b: &HostOrDeviceSlice2D<$field>) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_sub_arr_multi(a.len() as u32, a[0].len() as u32, a.as_mut_ptr(), b.as_ptr())
                        .wrap()
                }
            }

            fn mul_by_scalar_multi(a: &mut HostOrDeviceSlice2D<$field>, scalar: $field) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_mul_by_scalar_multi(a.len() as u32, a[0].len() as u32, a.as_mut_ptr(), scalar)
                        .wrap()
                }
            }
        }
    };
}
