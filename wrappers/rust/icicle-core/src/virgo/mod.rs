use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{error::IcicleResult, traits::FieldImpl};

pub trait Virgo<F: FieldImpl> {
    fn sumcheck_sum_unchecked(
        arr1: &HostOrDeviceSlice<F>,
        arr2: &HostOrDeviceSlice<F>,
        result: &mut HostOrDeviceSlice<F>,
        n: u32,
    ) -> IcicleResult<()>;
}

pub fn sumcheck_sum<F>(
    arr1: &HostOrDeviceSlice<F>,
    arr2: &HostOrDeviceSlice<F>,
    result: &mut HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Virgo<F>,
{
    <<F as FieldImpl>::Config as Virgo<F>>::sumcheck_sum_unchecked(arr1, arr2, result, n)
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
            use crate::virgo::{$field, $field_config, CudaError, DeviceContext};

            extern "C" {
                #[link_name = concat!($field_prefix, "SumcheckSum")]
                pub(crate) fn _sumcheck_sum(
                    arr1: *const $field,
                    arr2: *const $field,
                    result: *mut $field,
                    n: u32,
                ) -> CudaError;
            }
        }

        impl Virgo<$field> for $field_config {
            fn sumcheck_sum_unchecked(
                arr1: &HostOrDeviceSlice<$field>,
                arr2: &HostOrDeviceSlice<$field>,
                result: &mut HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_sumcheck_sum(arr1.as_ptr(), arr2.as_ptr(), result.as_mut_ptr(), n).wrap()
                }
            }
        }
    };
}
