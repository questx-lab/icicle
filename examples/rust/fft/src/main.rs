use ark_std::cfg_chunks;
use ark_std::cfg_into_iter;
use ark_std::cfg_iter;
use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::field::Field as IcicleField;
use icicle_core::traits::FieldConfig;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::MontgomeryConvertible;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::time::Instant;

use ark_ff::BigInt;
use ark_ff::BigInteger;
use ark_ff::Field;
use ark_ff::Fp;
use num_bigint::BigUint;

use icicle_core::fft::{fft_evaluate, fft_interpolate};

pub type ArkFrBN254 = ark_bn254::Fr;

#[cfg(feature = "profile")]
use std::time::Instant;

fn read_ws(name: &str, n: usize, size: usize) -> Vec<IcicleFrBN254> {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .open(name)
        .unwrap();

    let mut buf: Vec<u8> = vec![0; size * n];
    file.read(&mut buf)
        .unwrap();

    cfg_chunks!(buf, size)
        .map(|x| IcicleFrBN254::from_bytes_le(x))
        .collect()
}

pub fn run_fft(a: Vec<ArkFrBN254>) {
    let n = a.len();

    let start = Instant::now();

    let inout: Vec<IcicleFrBN254> = cfg_into_iter!(a)
        .map(|x| IcicleFrBN254::from(x.0 .0))
        .collect();

    println!("Finish conversion, loading time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let ws = read_ws("ws.bin", n, 32);
    let ws_inv = read_ws("ws_inv.bin", n, 32);

    println!("Finish loading ws! time = {:.2?}", start.elapsed());
    let start = Instant::now();

    // let mut ws_slice = HostOrDeviceSlice::on_host(ws);
    let mut ws_slice = HostOrDeviceSlice::cuda_malloc(ws.len()).unwrap();
    ws_slice
        .copy_from_host(&ws)
        .unwrap();

    // let mut ws_inv_slice = HostOrDeviceSlice::on_host(ws_inv);
    let mut ws_inv_slice = HostOrDeviceSlice::cuda_malloc(ws_inv.len()).unwrap();
    ws_inv_slice
        .copy_from_host(&ws_inv)
        .unwrap();

    let is_mont = true;

    println!("Done preparing. Start running on GPU... time = {:.2?}", start.elapsed());
    let start = Instant::now();

    // let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut inout_slice = HostOrDeviceSlice::cuda_malloc(inout.len()).unwrap();
    inout_slice
        .copy_from_host(&inout)
        .unwrap();

    fft_evaluate(&mut inout_slice, &mut ws_slice, n as u32, is_mont).unwrap();

    fft_interpolate(&mut inout_slice, &mut ws_inv_slice, n as u32, is_mont).unwrap();

    println!("Done Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let mut inout = vec![IcicleFrBN254::zero(); n];
    inout_slice
        .copy_to_host(&mut inout)
        .unwrap();

    println!("Copy to host, time = {:.2?}", start.elapsed());

    let start = Instant::now();
    let x: Vec<ArkFrBN254> = cfg_into_iter!(inout)
        .map(|x| Fp(BigInt(x.limbs), std::marker::PhantomData))
        .collect();
    println!("{:?}", &x[0..8]);

    println!(
        "Done Convert back to Arkwork, time = {:.2?} {}",
        start.elapsed(),
        x.len()
    );
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let n = (1 << 23) as usize;

    // let a: Vec<u128> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let mut a: Vec<ArkFrBN254> = Vec::with_capacity(n);
    for i in 0..n {
        let num = rng.gen_range(0..100);
        a.push(ArkFrBN254::from(num));

        if i < 8 {
            println!("num = {:?}", num);
        }
    }

    run_fft(a);
}
