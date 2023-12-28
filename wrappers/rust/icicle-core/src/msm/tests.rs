use super::MSM;
use crate::curve::{Affine, CurveConfig, Projective};
use crate::field::FieldConfig;
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_cuda_runtime::{memory::DeviceSlice, stream::CudaStream};

#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::VariableBaseMSM;
#[cfg(feature = "arkworks")]
use ark_std::{rand::Rng, test_rng, UniformRand};

pub fn check_msm<C: CurveConfig + MSM<C>, ScalarConfig: FieldConfig>()
where
    ScalarConfig: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1000, 1 << 18];
    let mut msm_results = DeviceSlice::cuda_malloc(1).unwrap();
    for test_size in test_sizes {
        let points = C::generate_random_affine_points(test_size);
        let scalars = ScalarConfig::generate_random(test_size);
        let points_ark: Vec<_> = points
            .iter()
            .map(|x| x.to_ark())
            .collect();
        let scalars_ark: Vec<_> = scalars
            .iter()
            .map(|x| x.to_ark())
            .collect();
        // if we simply transmute arkworks types, we'll get scalars or points in Montgomery format
        // (just beware the possible extra bit in affine point types, can't transmute ark Affine because of that)
        let scalars_mont = unsafe { &*(&scalars_ark[..] as *const _ as *const [C::ScalarField]) };

        let mut scalars_d = DeviceSlice::cuda_malloc(test_size).unwrap();
        let stream = CudaStream::create().unwrap();
        scalars_d
            .copy_from_host_async(&scalars_mont, &stream)
            .unwrap();

        let mut cfg = C::get_default_msm_config();
        cfg.ctx
            .stream = &stream;
        cfg.is_async = true;
        cfg.are_results_on_device = true;
        cfg.are_scalars_on_device = true;
        cfg.are_scalars_montgomery_form = true;
        C::msm(&scalars_d.as_slice(), &points, &cfg, &mut msm_results.as_slice()).unwrap();

        let mut msm_host_result = vec![Projective::<C>::zero(); 1];
        msm_results
            .copy_to_host(&mut msm_host_result[..])
            .unwrap();
        stream
            .synchronize()
            .unwrap();
        stream
            .destroy()
            .unwrap();

        let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
            VariableBaseMSM::msm(&points_ark, &scalars_ark).unwrap();
        let msm_res_affine: ark_ec::short_weierstrass::Affine<C::ArkSWConfig> = msm_host_result[0]
            .to_ark()
            .into();
        assert!(msm_res_affine.is_on_curve());
        assert_eq!(msm_host_result[0].to_ark(), msm_result_ark);
    }
}

pub fn check_msm_batch<C: CurveConfig + MSM<C>, ScalarConfig: FieldConfig>()
where
    ScalarConfig: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1000, 1 << 16];
    let batch_sizes = [1, 3, 1 << 4];
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = C::generate_random_affine_points(test_size);
            let scalars = ScalarConfig::generate_random(test_size * batch_size);

            let mut msm_results_1 = DeviceSlice::cuda_malloc(batch_size).unwrap();
            let mut msm_results_2 = DeviceSlice::cuda_malloc(batch_size).unwrap();
            let mut points_d = DeviceSlice::cuda_malloc(test_size * batch_size).unwrap();
            // a version of batched msm without using `cfg.points_size`, requires copying bases
            let points_cloned: Vec<Affine<C>> = std::iter::repeat(points.clone())
                .take(batch_size)
                .flatten()
                .collect();
            let stream = CudaStream::create().unwrap();
            points_d
                .copy_from_host_async(&points_cloned, &stream)
                .unwrap();

            let mut cfg = C::get_default_msm_config();
            cfg.ctx
                .stream = &stream;
            cfg.batch_size = batch_size as i32;
            cfg.is_async = true;
            cfg.are_results_on_device = true;
            C::msm(&scalars, &points, &cfg, &mut msm_results_1.as_slice()).unwrap();
            cfg.points_size = (test_size * batch_size) as i32;
            cfg.are_points_on_device = true;
            C::msm(&scalars, &points_d.as_slice(), &cfg, &mut msm_results_2.as_slice()).unwrap();

            let mut msm_host_result_1 = vec![Projective::<C>::zero(); batch_size];
            let mut msm_host_result_2 = vec![Projective::<C>::zero(); batch_size];
            msm_results_1
                .copy_to_host_async(&mut msm_host_result_1[..], &stream)
                .unwrap();
            msm_results_2
                .copy_to_host_async(&mut msm_host_result_2[..], &stream)
                .unwrap();
            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            let points_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();
            for (i, scalars_chunk) in scalars_ark
                .chunks(test_size)
                .enumerate()
            {
                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_ark, &scalars_chunk).unwrap();
                assert_eq!(msm_host_result_1[i].to_ark(), msm_result_ark);
                assert_eq!(msm_host_result_2[i].to_ark(), msm_result_ark);
            }
        }
    }
}

pub fn check_msm_skewed_distributions<C: CurveConfig + MSM<C>, ScalarConfig: FieldConfig>()
where
    ScalarConfig: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1 << 6, 1000];
    let batch_sizes = [1, 3, 1 << 8];
    let rng = &mut test_rng();
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = C::generate_random_affine_points(test_size * batch_size);
            let mut scalars = vec![C::ScalarField::zero(); test_size * batch_size];
            for _ in 0..(test_size * batch_size / 2) {
                scalars[rng.gen_range(0..test_size * batch_size)] = C::ScalarField::one();
            }
            for _ in (1 << 8)..test_size {
                scalars[rng.gen_range(0..test_size * batch_size)] =
                    C::ScalarField::from_ark(<C::ScalarField as ArkConvertible>::ArkEquivalent::rand(rng));
            }

            let mut msm_results = vec![Projective::<C>::zero(); batch_size];

            let mut cfg = C::get_default_msm_config();
            cfg.batch_size = batch_size as i32;
            cfg.points_size = (test_size * batch_size) as i32;
            C::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();

            let points_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();
            for (i, (scalars_chunk, points_chunk)) in scalars_ark
                .chunks(test_size)
                .zip(points_ark.chunks(test_size))
                .enumerate()
            {
                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_chunk, &scalars_chunk).unwrap();
                assert_eq!(msm_results[i].to_ark(), msm_result_ark);
            }
        }
    }
}
