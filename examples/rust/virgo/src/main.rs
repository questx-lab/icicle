use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use rand::Rng;
use std::str::FromStr;
use std::time::Instant;

use ark_ff::PrimeField;
use ark_std::cfg_into_iter;
use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::virgo::sumcheck_sum;
pub type ArkFrBN254 = ark_bn254::Fr;

fn run_sumcheck_sum(arr1: Vec<ArkFrBN254>, arr2: Vec<ArkFrBN254>) {
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
    let start = Instant::now();

    println!("START running on GPU");
    let start = Instant::now();
    _ = sumcheck_sum(&a_slice, &b_slice, &mut result_slice, n as u32);
    println!("DONE Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let mut result = vec![IcicleFrBN254::zero(); 1];
    result_slice
        .copy_to_host(&mut result)
        .unwrap();

    // double check with the result on cpu
    let mut cpu_sum = ArkFrBN254::from(0u128);
    for i in 0..arr1.len() {
        cpu_sum += arr1[i] * arr2[i];
    }

    assert_eq!(
        IcicleFrBN254::from(
            cpu_sum
                .into_bigint()
                .0,
        ),
        result[0]
    );

    println!("Test passed!");
}

fn main() {
    let n = 1 << 16;
    let mut a: Vec<ArkFrBN254> = Vec::with_capacity(n);
    let mut b: Vec<ArkFrBN254> = Vec::with_capacity(n);

    for i in 0..n {
        // let mut rng = StdRng::seed_from_u64(42);
        let mut rng = rand::thread_rng();

        let num = rng.gen_range(0..1000);
        a.push(ArkFrBN254::from(num));

        let num = rng.gen_range(0..1000);
        b.push(ArkFrBN254::from(num));
    }

    run_sumcheck_sum(a, b);
}
