use ark_ff::BigInt;
use ark_ff::Fp;
use ark_ff::PrimeField;
use ark_std::cfg_into_iter;

use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::traits::FieldImpl;
use icicle_core::virgo::bk_sum_all_case1;
use icicle_core::virgo::bk_sum_all_case2;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::str::FromStr;
use std::time::Instant;

pub type ArkFrBN254 = ark_bn254::Fr;

fn run_bk_sum_all_case1(arr1: Vec<ArkFrBN254>, arr2: Vec<ArkFrBN254>) {
    let n = arr1.len();
    let mut a_slice = HostOrDeviceSlice::cuda_malloc(n).unwrap();
    let mut b_slice = HostOrDeviceSlice::cuda_malloc(n).unwrap();

    println!("Convert and copy data...");
    let start = Instant::now();

    let a: Vec<IcicleFrBN254> = cfg_into_iter!(&arr1)
        .map(|x| IcicleFrBN254::from(x.0 .0))
        .collect();
    let b: Vec<IcicleFrBN254> = cfg_into_iter!(&arr2)
        .map(|x| IcicleFrBN254::from(x.0 .0))
        .collect();

    a_slice
        .copy_from_host(&a)
        .unwrap();

    b_slice
        .copy_from_host(&b)
        .unwrap();

    let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    println!("DONE Convert and copy to device, time = {:.2?}", start.elapsed());

    println!("START running on GPU");
    let start = Instant::now();
    _ = bk_sum_all_case1(&a_slice, &b_slice, &mut result_slice, n as u32);
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

    println!("Test passed!");
}

// Bookkeeping sum_all test 2
fn run_bk_sum_all_case2(arr: Vec<ArkFrBN254>) {
    let n = arr.len();
    let mut a_slice = HostOrDeviceSlice::cuda_malloc(n).unwrap();

    println!("Convert and copy data...");
    let start = Instant::now();

    let a: Vec<IcicleFrBN254> = cfg_into_iter!(&arr)
        .map(|x| IcicleFrBN254::from(x.0 .0))
        .collect();

    a_slice
        .copy_from_host(&a)
        .unwrap();

    let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    println!("DONE Convert and copy to device, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    println!("START running on GPU");
    let start = Instant::now();
    _ = bk_sum_all_case2(&a_slice, &mut result_slice, n as u32);
    println!("DONE Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let mut result_mont = vec![IcicleFrBN254::zero(); 1];
    result_slice
        .copy_to_host(&mut result_mont)
        .unwrap();

    let result: ArkFrBN254 = Fp(BigInt(result_mont[0].limbs), std::marker::PhantomData);

    // double check with the result on cpu
    let mut cpu_sum = ArkFrBN254::from(0u128);
    for i in 0..arr.len() {
        cpu_sum += arr[i];
    }

    assert_eq!(cpu_sum, result);

    println!("Test passed!");
}

fn main() {
    let n = 1 << 2;
    let mut a: Vec<ArkFrBN254> = Vec::with_capacity(n);
    let mut b: Vec<ArkFrBN254> = Vec::with_capacity(n);

    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..n {
        // let mut rng = rand::thread_rng();

        let num = rng.gen_range(0..10);
        a.push(ArkFrBN254::from(num));

        let num = rng.gen_range(0..10);
        b.push(ArkFrBN254::from(num));
    }

    run_bk_sum_all_case1(a, b);
    // run_bk_sum_all_case2(a);
}
