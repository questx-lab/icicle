use ark_ff::BigInt;
use ark_ff::Fp;
use ark_ff::PrimeField;
use ark_std::cfg_into_iter;

use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::traits::FieldImpl;
use icicle_core::virgo::bk_produce_case_1;
use icicle_core::virgo::bk_produce_case_2;
use icicle_core::virgo::bk_sum_all_case_1;
use icicle_core::virgo::bk_sum_all_case_2;
use icicle_core::virgo::VirgoConfig;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::memory::HostOrDeviceSlice::Device;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::str::FromStr;
use std::time::Instant;

pub type ArkFrBN254 = ark_bn254::Fr;

fn arks_to_icicles(arr1: &Vec<ArkFrBN254>) -> HostOrDeviceSlice<'static, IcicleFrBN254> {
    let n = arr1.len();
    let mut a_slice = HostOrDeviceSlice::cuda_malloc(n).unwrap();
    let a: Vec<IcicleFrBN254> = cfg_into_iter!(&arr1)
        .map(|x| IcicleFrBN254::from(x.0 .0))
        .collect();
    a_slice
        .copy_from_host(&a)
        .unwrap();

    a_slice
}

fn icicles_to_arks(result_slice: HostOrDeviceSlice<'static, IcicleFrBN254>, n: usize) -> Vec<ArkFrBN254> {
    let mut result_mont = vec![IcicleFrBN254::zero(); n];
    result_slice
        .copy_to_host(&mut result_mont)
        .unwrap();

    let result: Vec<ArkFrBN254> = result_mont
        .iter()
        .map(|x| Fp(BigInt(x.limbs), std::marker::PhantomData))
        .collect();

    result
}

fn run_bk_sum_all_case_1(arr1: Vec<ArkFrBN254>, arr2: Vec<ArkFrBN254>) {
    let n = arr1.len();
    let mut a_slice = arks_to_icicles(&arr1);
    let mut b_slice = arks_to_icicles(&arr2);

    let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();

    println!("START running on GPU");
    let start = Instant::now();
    _ = bk_sum_all_case_1(&VirgoConfig::default(), &a_slice, &b_slice, &mut result_slice, n as u32);
    println!("DONE Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let mut result_mont = vec![IcicleFrBN254::zero(); 1];
    result_slice
        .copy_to_host(&mut result_mont)
        .unwrap();

    let result: ArkFrBN254 = Fp(BigInt(result_mont[0].limbs), std::marker::PhantomData);

    // double check with the result on cpu
    let mut cpu_sum = ArkFrBN254::from(0u128);
    for i in 0..arr1.len() {
        cpu_sum += arr1[i] * arr2[i];
    }

    assert_eq!(cpu_sum, result);

    println!("Test run_bk_sum_all_case_1 passed!");
}

// Bookkeeping sum_all test 2
fn run_bk_sum_all_case_2(arr: Vec<ArkFrBN254>) {
    let n = arr.len();
    let mut a_slice = arks_to_icicles(&arr);

    let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();

    println!("START running on GPU");
    let start = Instant::now();
    _ = bk_sum_all_case_2(&VirgoConfig::default(), &a_slice, &mut result_slice, n as u32);
    println!("DONE Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let result = icicles_to_arks(result_slice, 1)[0];

    // double check with the result on cpu
    let mut cpu_sum = ArkFrBN254::from(0u128);
    for i in 0..arr.len() {
        cpu_sum += arr[i];
    }

    assert_eq!(cpu_sum, result);

    println!("Test run_bk_sum_all_case_2 passed!");
}

fn run_bk_produce_case_1(arr1: Vec<ArkFrBN254>, arr2: Vec<ArkFrBN254>) {
    let n = arr1.len();
    let start = Instant::now();
    let mut a_slice = arks_to_icicles(&arr1);
    let mut b_slice = arks_to_icicles(&arr2);
    let mut result_slice = HostOrDeviceSlice::cuda_malloc(3).unwrap();
    println!("Copy CPU -> GPU: time = {:.2?}", start.elapsed());

    _ = bk_produce_case_1(&VirgoConfig::default(), &a_slice, &b_slice, &mut result_slice, n as u32);

    let result = icicles_to_arks(result_slice, 3);

    // double check with the result on cpu
    let mut cpu_sum = vec![ArkFrBN254::from(0u128); 3];
    let two = ArkFrBN254::from(2u128);
    for i0 in (0..n).step_by(2) {
        let i1 = i0 + 1;
        cpu_sum[0] += arr1[i0] * arr2[i0];
        cpu_sum[1] += arr1[i1] * arr2[i1];
        cpu_sum[2] += (two * arr1[i1] - arr1[i0]) * (two * arr2[i1] - arr2[i0]);
    }

    assert_eq!(cpu_sum, result);

    println!("Test passed!");
}

fn run_bk_produce_case_2(arr: Vec<ArkFrBN254>) {
    let n = arr.len();
    let start = Instant::now();
    let mut a_slice = arks_to_icicles(&arr);
    let mut result_slice = HostOrDeviceSlice::cuda_malloc(3).unwrap();
    println!("Copy CPU -> GPU: time = {:.2?}", start.elapsed());

    _ = bk_produce_case_2(&VirgoConfig::default(), &a_slice, &mut result_slice, n as u32);

    let result = icicles_to_arks(result_slice, 3);

    // double check with the result on cpu
    let mut cpu_sum = vec![ArkFrBN254::from(0u128); 3];
    let two = ArkFrBN254::from(2u128);
    for i0 in (0..n).step_by(2) {
        let i1 = i0 + 1;
        cpu_sum[0] += arr[i0];
        cpu_sum[1] += arr[i1];
        cpu_sum[2] += two * arr[i1] - arr[i0];
    }

    assert_eq!(cpu_sum, result);

    println!("Test passed!");
}

fn main() {
    let n = 1 << 20;
    let mut a: Vec<ArkFrBN254> = Vec::with_capacity(n);
    let mut b: Vec<ArkFrBN254> = Vec::with_capacity(n);

    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..n {
        let num = rng.gen_range(0..10);
        a.push(ArkFrBN254::from(num));

        let num = rng.gen_range(0..10);
        b.push(ArkFrBN254::from(num));

        // println!(
        //     "a = {}, b = {}\n",
        //     a[a.len() - 1].to_string(),
        //     b[b.len() - 1].to_string()
        // );
    }

    // run_bk_sum_all_case_1(a, b);
    // run_bk_sum_all_case_2(a);
    // run_bk_produce_case_1(a, b);
    run_bk_produce_case_2(a);
}
