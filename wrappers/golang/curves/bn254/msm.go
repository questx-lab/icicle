package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254

import "C"

import (
	"local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
)

func GetDefaultMSMConfig() core.MSMConfig {
	return C.BN254DefaultMSMConfig()
}

func Msm[T cr.HostOrDeviceSlice](scalars T,	points T, cfg *core.MSMConfig, results T) {
	// if len(scalars) % len(points) != 0 {
	// 	panic(
	// 			"Number of points {} does not divide the number of scalars {}",
	// 			points.len(),
	// 			scalars.len(),
	// 	);
	// }
	// if scalars.len() % results.len() != 0 {
	// 		panic(
	// 				"Number of results {} does not divide the number of scalars {}",
	// 				results.len(),
	// 				scalars.len(),
	// 		);
	// }
	// let mut local_cfg = cfg.clone();
	// local_cfg.points_size = points.len() as i32;
	// local_cfg.batch_size = results.len() as i32;
	// local_cfg.are_scalars_on_device = scalars.is_on_device();
	// local_cfg.are_points_on_device = points.is_on_device();
	// local_cfg.are_results_on_device = results.is_on_device();

	// C::msm_unchecked(scalars, points, &local_cfg, results)
	C.BN254MSMCuda(scalars, points, scalars.len() / results.len(), cfg, results)
}