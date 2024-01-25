package cuda_runtime

import (
	"local/hello/icicle/wrappers/golang/cuda_runtime/cuda_bindings"
)

type Stream struct {
	handle *cuda_bindings.CudaStream
}

func CreateStream() (Stream, cuda_bindings.CudaErrorT) {
	var stream cuda_bindings.CudaStream
	error := cuda_bindings.CudaStreamCreate(&stream)
	return Stream{handle: &stream}, error
}

func CreateStreamWithFlags(flags uint32) (Stream, cuda_bindings.CudaErrorT) {
	var stream cuda_bindings.CudaStream
	error := cuda_bindings.CudaStreamCreateWithFlags(&stream, flags)
	return Stream{handle: &stream}, error
}

func (s* Stream) DestroyStream() cuda_bindings.CudaErrorT {
	err := cuda_bindings.CudaStreamDestroy(*s.handle)
	if err == cuda_bindings.CudaSuccess {
		s.handle = nil
	}
	return err
}

func (s* Stream) SynchronizeStream() cuda_bindings.CudaErrorT {
	return cuda_bindings.CudaStreamSynchronize(*s.handle)
}

// TODO:
// func CudaStreamWaitEvent(Stream CudaStream, Event CudaEvent, Flags uint32) CudaErrorT
// func CudaStreamQuery(Stream CudaStream) CudaErrorT
