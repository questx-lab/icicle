package core

import (
	"local/hello/icicle/wrappers/golang/cuda_runtime/cuda_bindings"
)

type IcicleErrorCode int 

const (
	IcicleSuccess 					IcicleErrorCode = 0
	InvalidArgument 				IcicleErrorCode = 1
	MemoryAllocationError 	IcicleErrorCode = 2
	InternalCudaError				IcicleErrorCode = 199999999
	UndefinedError 					IcicleErrorCode = 999999999
)

type IcicleError struct {
	IcicleErrorCode IcicleErrorCode
	CudaErrorCode	cuda_bindings.CudaErrorT
	reason string
}

func (ie* IcicleError) FromCudaError(error cuda_bindings.CudaErrorT) IcicleError {
	switch (error) {
	case cuda_bindings.CudaSuccess:
		ie.IcicleErrorCode = IcicleSuccess
	default:
		ie.IcicleErrorCode = InternalCudaError
	}

	ie.CudaErrorCode = error
	ie.reason = "Runtime CUDA error."

	return *ie
}

func FromCodeAndReason(code IcicleErrorCode, reason string) IcicleError {
	return IcicleError {
		IcicleErrorCode: code,
		reason: reason,
		CudaErrorCode: cuda_bindings.CudaErrorUnknown,
	}
}