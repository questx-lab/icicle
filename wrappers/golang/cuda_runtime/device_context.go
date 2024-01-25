package cuda_runtime

import (
	"local/hello/icicle/wrappers/golang/cuda_runtime/cuda_bindings"
)

type DeviceContext struct {
	/// Stream to use. Default value: 0.
	Stream *cuda_bindings.CudaStream // Assuming the type is provided by a CUDA binding crate

	/// Index of the currently used GPU. Default value: 0.
	DeviceId uint

	/// Mempool to use. Default value: 0.
	Mempool CudaMemPool // Assuming the type is provided by a CUDA binding crate
}

func GetDefaultDeviceContext() DeviceContext {
	var defaultStream cuda_bindings.CudaStream
	defaultStream.CreateStream()
	return DeviceContext {
			&defaultStream,
			0,
			0,
	}
}