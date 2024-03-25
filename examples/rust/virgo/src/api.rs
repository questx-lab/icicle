#[cfg(test)]
mod test {
    use crate::constant::ArkFrBN254;
    use crate::constant::D;
    use crate::constant::K_BN254;

    use ark_ff::BigInt;
    use ark_ff::Fp;
    use ark_std::cfg_into_iter;

    use icicle_bn254::curve::ScalarField as IcicleFrBN254;
    use icicle_core::traits::FieldImpl;
    use icicle_core::virgo::bk_produce_case_1;
    use icicle_core::virgo::bk_produce_case_2;
    use icicle_core::virgo::bk_sum_all_case_1;
    use icicle_core::virgo::bk_sum_all_case_2;
    use icicle_core::virgo::build_merkle_tree;
    use icicle_core::virgo::hash_merkle_tree_slice;
    use icicle_core::virgo::MerkleTreeConfig;
    use icicle_core::virgo::SumcheckConfig;
    use icicle_cuda_runtime::memory::HostOrDeviceSlice;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::str::FromStr;
    use std::time::Instant;

    fn gen_input() -> (Vec<ArkFrBN254>, Vec<ArkFrBN254>) {
        let n = 1 << 21;
        let mut a: Vec<ArkFrBN254> = Vec::with_capacity(n);
        let mut b: Vec<ArkFrBN254> = Vec::with_capacity(n);

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..n {
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

        (a, b)
    }

    fn arks_to_icicles_device(arr1: &Vec<ArkFrBN254>) -> HostOrDeviceSlice<'static, IcicleFrBN254> {
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

    fn u32s_to_device(arr1: &Vec<u32>) -> HostOrDeviceSlice<'static, u32> {
        let mut a_slice = HostOrDeviceSlice::cuda_malloc(arr1.len()).unwrap();
        a_slice
            .copy_from_host(&arr1)
            .unwrap();

        a_slice
    }

    fn icicles_to_arks(result_slice: &HostOrDeviceSlice<'static, IcicleFrBN254>, n: usize) -> Vec<ArkFrBN254> {
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

    #[test]
    fn test_bk_sum_all_case_1() {
        let (arr1, arr2) = gen_input();
        let n = arr1.len();
        let a_slice = arks_to_icicles_device(&arr1);
        let b_slice = arks_to_icicles_device(&arr2);

        let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();

        println!("START running on GPU");
        let start = Instant::now();
        _ = bk_sum_all_case_1(
            &SumcheckConfig::default(),
            &a_slice,
            &b_slice,
            &mut result_slice,
            n as u32,
        );
        println!("DONE Running on GPU, time = {:.2?}", start.elapsed());

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
    #[test]
    fn test_bk_sum_all_case_2() {
        let arr = gen_input().0;
        let n = arr.len();
        let a_slice = arks_to_icicles_device(&arr);

        let mut result_slice = HostOrDeviceSlice::cuda_malloc(1).unwrap();

        println!("START running on GPU");
        let start = Instant::now();
        _ = bk_sum_all_case_2(&SumcheckConfig::default(), &a_slice, &mut result_slice, n as u32);
        println!("DONE Running on GPU, time = {:.2?}", start.elapsed());

        let result = icicles_to_arks(&result_slice, 1)[0];

        // double check with the result on cpu
        let mut cpu_sum = ArkFrBN254::from(0u128);
        for i in 0..arr.len() {
            cpu_sum += arr[i];
        }

        assert_eq!(cpu_sum, result);

        println!("Test run_bk_sum_all_case_2 passed!");
    }

    #[test]
    fn test_bk_produce_case_1() {
        let (arr1, arr2) = gen_input();
        let n = arr1.len();
        let start = Instant::now();
        let a_slice = arks_to_icicles_device(&arr1);
        let b_slice = arks_to_icicles_device(&arr2);
        let mut result_slice = HostOrDeviceSlice::cuda_malloc(3).unwrap();
        println!("Copy CPU -> GPU: time = {:.2?}", start.elapsed());

        _ = bk_produce_case_1(
            &SumcheckConfig::default(),
            &a_slice,
            &b_slice,
            &mut result_slice,
            n as u32,
        );

        let result = icicles_to_arks(&result_slice, 3);

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

    #[test]
    fn test_bk_produce_case_2() {
        let arr = gen_input().0;
        let n = arr.len();
        let start = Instant::now();
        let a_slice = arks_to_icicles_device(&arr);
        let mut result_slice = HostOrDeviceSlice::cuda_malloc(3).unwrap();
        println!("Copy CPU -> GPU: time = {:.2?}", start.elapsed());

        _ = bk_produce_case_2(&SumcheckConfig::default(), &a_slice, &mut result_slice, n as u32);

        let result = icicles_to_arks(&result_slice, 3);

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

    fn get_merkle_tree(
        input: &Vec<ArkFrBN254>,
    ) -> (
        MerkleTreeConfig<IcicleFrBN254>,
        HostOrDeviceSlice<'static, IcicleFrBN254>,
    ) {
        let n = input.len();
        let mut tree_slice = HostOrDeviceSlice::cuda_malloc(2 * n - 1).unwrap();
        let a: Vec<IcicleFrBN254> = cfg_into_iter!(&input)
            .map(|x| IcicleFrBN254::from(x.0 .0))
            .collect();
        tree_slice
            .copy_from_host_partially(&a)
            .unwrap();

        let params = K_BN254.to_vec();

        let device_d = u32s_to_device(&D.to_vec());
        let device_params = arks_to_icicles_device(&params);
        let device_config = MerkleTreeConfig::default_for_device(&device_params, &device_d);

        (device_config, tree_slice)
    }

    fn get_device_value_at_index(tree_slice: &HostOrDeviceSlice<IcicleFrBN254>, index: usize) -> ArkFrBN254 {
        let mut ice_values = vec![IcicleFrBN254::zero(); 1];
        tree_slice
            .copy_to_host_at_index(&mut ice_values, 0, index)
            .unwrap();

        Fp(BigInt(ice_values[0].limbs), std::marker::PhantomData)
    }

    fn get_merkle_path(
        tree_slice: &HostOrDeviceSlice<IcicleFrBN254>,
        n: usize,
        index: usize,
    ) -> (ArkFrBN254, Vec<ArkFrBN254>) {
        let mut ice_result = vec![];

        let leave_value = get_device_value_at_index(tree_slice, index);

        // Get the path
        let mut x_n = n;
        let mut x_index = index;
        let mut offset = 0;
        loop {
            if x_n == 1 {
                ice_result.push(get_device_value_at_index(tree_slice, offset + x_index));
            } else {
                if x_index % 2 == 0 {
                    ice_result.push(get_device_value_at_index(tree_slice, offset + x_index + 1));
                } else {
                    ice_result.push(get_device_value_at_index(tree_slice, offset + x_index - 1));
                }
            }

            offset += x_n;
            x_index = x_index / 2;
            x_n /= 2;
            if x_n == 0 {
                break;
            }
        }

        (leave_value, ice_result)
    }

    #[test]
    fn test_build_merkle_tree() {
        // let input = gen_input().0;
        let input = vec![
            ArkFrBN254::from(1),
            ArkFrBN254::from(2),
            ArkFrBN254::from(3),
            ArkFrBN254::from(4),
            ArkFrBN254::from(5),
            ArkFrBN254::from(6),
            ArkFrBN254::from(7),
            ArkFrBN254::from(8),
        ];

        let (device_config, mut tree_slice) = get_merkle_tree(&input);
        build_merkle_tree(&device_config, &mut tree_slice, input.len() as u32).unwrap();

        let n = input.len();
        let result = icicles_to_arks(&tree_slice, 2 * n - 1)[n..].to_vec();

        let expected = vec![
            ArkFrBN254::from_str("2125786076286291193686112931062780544355053628865661388448738299372101689918")
                .unwrap(),
            ArkFrBN254::from_str("7566809574148433254186606897930770637848696490879826179200753398937677504597")
                .unwrap(),
            ArkFrBN254::from_str("6723075676534364259334340678583760616904048274740060999234670172267306587926")
                .unwrap(),
            ArkFrBN254::from_str("21498508090997097154125611486167291905316593418498458053376396031068091751763")
                .unwrap(),
            ArkFrBN254::from_str("11257601743655441955364798273076808917833807565662665471407412214232759496044")
                .unwrap(),
            ArkFrBN254::from_str("2458844257376397839950928938995518455050045729085527518555442443771609467092")
                .unwrap(),
            ArkFrBN254::from_str("20650870930010328676568597081132406095907917425495461688177797913895684422544")
                .unwrap(),
        ];
        assert_eq!(expected, result);

        for index in 0..input.len() {
            let (leave_value, ice_result) = get_merkle_path(&tree_slice, n, index);
            assert_eq!(input[index], leave_value);

            let log_n = input
                .len()
                .ilog2() as usize;
            assert_eq!(log_n + 1, ice_result.len());
            if index % 2 == 0 {
                assert_eq!(input[index + 1], ice_result[0]);
            } else {
                assert_eq!(input[index - 1], ice_result[0]);
            }
            let mut len = input.len() / 2;
            let mut x = index / 2;
            let mut offset = 0;

            for i in 0..log_n {
                if i == log_n - 1 {
                    // This must the be root of the Merkle tree
                    assert_eq!(expected[n - 2], ice_result[i + 1]);
                } else {
                    let cur_index = offset + x;
                    if cur_index % 2 == 0 {
                        assert_eq!(expected[cur_index + 1], ice_result[i + 1]);
                    } else {
                        assert_eq!(expected[cur_index - 1], ice_result[i + 1]);
                    }
                }
                offset += len;
                len /= 2;
                x /= 2;
            }
        }
    }

    #[test]
    fn test_hash_merkle_tree_slice() {
        let input = vec![
            ArkFrBN254::from(1),
            ArkFrBN254::from(2),
            ArkFrBN254::from(3),
            ArkFrBN254::from(4),
            ArkFrBN254::from(5),
            ArkFrBN254::from(6),
            ArkFrBN254::from(7),
            ArkFrBN254::from(8),
        ];

        let n = input.len();
        let slice_size = 2usize;
        let slice_count = n / slice_size;

        let (device_config, mut tree_slice) = get_merkle_tree(&input);
        let mut output_slice = HostOrDeviceSlice::cuda_malloc(slice_count).unwrap();

        hash_merkle_tree_slice(
            &device_config,
            &mut tree_slice,
            &mut output_slice,
            n as u32,
            slice_size as u32,
        )
        .unwrap();

        let actual = icicles_to_arks(&output_slice, slice_count);
        let expected = vec![
            ArkFrBN254::from_str("2125786076286291193686112931062780544355053628865661388448738299372101689918")
                .unwrap(),
            ArkFrBN254::from_str("7566809574148433254186606897930770637848696490879826179200753398937677504597")
                .unwrap(),
            ArkFrBN254::from_str("6723075676534364259334340678583760616904048274740060999234670172267306587926")
                .unwrap(),
            ArkFrBN254::from_str("21498508090997097154125611486167291905316593418498458053376396031068091751763")
                .unwrap(),
        ];
        assert_eq!(expected, actual);
    }
}
