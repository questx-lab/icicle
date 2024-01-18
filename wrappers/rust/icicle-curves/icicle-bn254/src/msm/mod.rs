use crate::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    impl_msm,
    msm::{MSMConfig, MSM},
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::{error::CudaError, memory::HostOrDeviceSlice};

impl_msm!("bn254", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use std::{fs::File, io::Read};

    use icicle_core::impl_msm_tests;
    use icicle_core::msm::msm;
    use icicle_core::msm::tests::*;
    use icicle_core::msm::MSM;
    use icicle_cuda_runtime::memory::HostOrDeviceSlice;

    use crate::curve::G1Projective;
    use crate::curve::{CurveCfg, G1Affine, ScalarField};

    impl_msm_tests!(CurveCfg);

    #[test]
    fn test_msm_bug() {
        let mut file = File::open("scalars.bin").unwrap();
        let mut scalars = Vec::new();
        let mut buffer = [0; std::mem::size_of::<ScalarField>()];
        while let Ok(_) = file.read_exact(&mut buffer) {
            let item: ScalarField = unsafe { std::ptr::read(buffer.as_ptr() as *const _) };
            scalars.push(item);
        }

        let mut file = File::open("bases.bin").unwrap();
        let mut bases = Vec::new();
        let mut buffer = [0; std::mem::size_of::<G1Affine>()];
        while let Ok(_) = file.read_exact(&mut buffer) {
            let item: G1Affine = unsafe { std::ptr::read(buffer.as_ptr() as *const _) };
            bases.push(item);
        }
    
        // let push_n = (1 << 17) - scalars.len();
        // for _ in 0..push_n {
        //     scalars.push(ScalarField::zero());
        //     bases.push(G1Affine::zero());
        // }
        // println!("Len now: {}", scalars.len());

        let mut msm_config = CurveCfg::get_default_msm_config();
        msm_config.are_scalars_montgomery_form = true;
        msm_config.are_points_montgomery_form = true;

        let scalars_buffer = HostOrDeviceSlice::on_host(scalars);
        let bases_buffer = HostOrDeviceSlice::on_host(bases);

        let mut result = HostOrDeviceSlice::on_host(vec![G1Projective::zero()]);
        msm(
            &scalars_buffer,
            &bases_buffer,
            &msm_config,
            &mut result,
        )
        .unwrap();
        let result = G1Affine::from(result[0..1][0]);
        println!("Result: {:?}", result);

        println!("Correct result should be {{ x: 0x2b8cc00995ba19bbbd97ae67ba3e9d88ba57a983bcdb54530b4dc97c63140de9, y: 0x23d8581d74b232c5d1ad7ac225fefd3cecea05190c7413da98b2cec46dbc2192 }}");
    }
}
