use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::{error::IcicleResult, traits::FieldImpl};

pub trait Fft<F: FieldImpl> {
    fn evaluate_unchecked(
        inout: &mut HostOrDeviceSlice<F>,
        ws: &HostOrDeviceSlice<F>,
        n: u32,
        is_montgomery: bool,
    ) -> IcicleResult<()>;

    fn interpolate_unchecked(
        inout: &mut HostOrDeviceSlice<F>,
        ws: &HostOrDeviceSlice<F>,
        n: u32,
        is_montgomery: bool,
    ) -> IcicleResult<()>;
}

pub fn fft_evaluate<F>(
    inout: &mut HostOrDeviceSlice<F>,
    ws: &HostOrDeviceSlice<F>,
    n: u32,
    is_montgomery: bool,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::evaluate_unchecked(inout, ws, n, is_montgomery)
}

pub fn fft_interpolate<F>(
    inout: &mut HostOrDeviceSlice<F>,
    ws: &HostOrDeviceSlice<F>,
    n: u32,
    is_montgomery: bool,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::interpolate_unchecked(inout, ws, n, is_montgomery)
}

#[macro_export]
macro_rules! impl_fft {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
      ) => {
        mod $field_prefix_ident {
            use crate::fft::{$field, $field_config, CudaError, DeviceContext};

            extern "C" {
                #[link_name = concat!($field_prefix, "FftEvaluate")]
                pub(crate) fn _fft_evaluate(
                    inout: *mut $field,
                    ws: *const $field,
                    n: u32,
                    is_montgomery: bool,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FftInterpolate")]
                pub(crate) fn _fft_interpolate(
                    inout: *mut $field,
                    ws: *const $field,
                    n: u32,
                    is_montgomery: bool,
                ) -> CudaError;
            }
        }

        impl Fft<$field> for $field_config {
            fn evaluate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
                is_montgomery: bool,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_fft_evaluate(inout.as_mut_ptr(), ws.as_ptr(), n, is_montgomery).wrap() }
            }

            fn interpolate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
                is_montgomery: bool,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_fft_interpolate(inout.as_mut_ptr(), ws.as_ptr(), n, is_montgomery).wrap()
                }
            }
        }
    };
}
