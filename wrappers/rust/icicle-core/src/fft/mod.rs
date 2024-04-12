use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostOrDeviceSlice2D};

use crate::{error::IcicleResult, traits::FieldImpl};

pub trait Fft<F: FieldImpl> {
    fn evaluate_unchecked(inout: &mut HostOrDeviceSlice<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;
    fn interpolate_unchecked(inout: &mut HostOrDeviceSlice<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;

    fn evaluate_multi(inout: &mut HostOrDeviceSlice2D<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;
    fn interpolate_multi(inout: &mut HostOrDeviceSlice2D<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>;
}

pub fn fft_evaluate<F>(inout: &mut HostOrDeviceSlice<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::evaluate_unchecked(inout, ws, n)
}

pub fn fft_interpolate<F>(inout: &mut HostOrDeviceSlice<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::interpolate_unchecked(inout, ws, n)
}

pub fn fft_evaluate_multi<F>(inout: &mut HostOrDeviceSlice2D<F>, ws: &HostOrDeviceSlice<F>, n: u32) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::evaluate_multi(inout, ws, n)
}

pub fn fft_interpolate_multi<F>(
    inout: &mut HostOrDeviceSlice2D<F>,
    ws: &HostOrDeviceSlice<F>,
    n: u32,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: Fft<F>,
{
    <<F as FieldImpl>::Config as Fft<F>>::interpolate_multi(inout, ws, n)
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
                pub(crate) fn _fft_evaluate(inout: *mut $field, ws: *const $field, n: u32) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FftInterpolate")]
                pub(crate) fn _fft_interpolate(inout: *mut $field, ws: *const $field, n: u32) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FftEvaluateMulti")]
                pub(crate) fn _fft_evaluate_multi(
                    num_fft: u32,
                    inout: *const *mut $field,
                    ws: *const $field,
                    n: u32,
                ) -> CudaError;
            }

            extern "C" {
                #[link_name = concat!($field_prefix, "FftInterpolateMulti")]
                pub(crate) fn _fft_interpolate_multi(
                    num_fft: u32,
                    inout: *const *mut $field,
                    ws: *const $field,
                    n: u32,
                ) -> CudaError;
            }
        }

        impl Fft<$field> for $field_config {
            fn evaluate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_fft_evaluate(inout.as_mut_ptr(), ws.as_ptr(), n).wrap() }
            }

            fn interpolate_unchecked(
                inout: &mut HostOrDeviceSlice<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe { $field_prefix_ident::_fft_interpolate(inout.as_mut_ptr(), ws.as_ptr(), n).wrap() }
            }

            fn evaluate_multi(
                inout: &mut HostOrDeviceSlice2D<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_fft_evaluate_multi(inout.len() as u32, inout.as_mut_ptr(), ws.as_ptr(), n)
                        .wrap()
                }
            }

            fn interpolate_multi(
                inout: &mut HostOrDeviceSlice2D<$field>,
                ws: &HostOrDeviceSlice<$field>,
                n: u32,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::_fft_interpolate_multi(inout.len() as u32, inout.as_mut_ptr(), ws.as_ptr(), n)
                        .wrap()
                }
            }
        }
    };
}
