package cuda_runtime

import "C"
import (
	"unsafe"
	"local/hello/icicle/wrappers/golang/cuda_runtime/cuda_bindings"
)

type HostOrDeviceSlice interface {
	Len() int
	Cap() int
	IsEmpty() bool
	AsSlice() []any
	AsPointer() *any
	IsOnDevice() bool
}

type DeviceSlice struct {
	inner unsafe.Pointer
	capacity int
	length int
}

func (d* DeviceSlice) Len() int {
	return d.length
}

func (d* DeviceSlice) Cap() int {
	return d.capacity
}

func (d* DeviceSlice) IsEmpty() bool {
	return d.length == 0
}

func (d* DeviceSlice) AsSlice() []any {
	panic("Use CopyToHost or CopyToHostAsync to move device data to a slice")
}

func (d* DeviceSlice) AsPointer() *any {
	return (*any)(d.inner)
}

func (d DeviceSlice) IsOnDevice() bool {
	return true
}

type HostSlice struct {
	inner	[]any
}

func (h* HostSlice) Len() int {
	return len(h.inner)
}

func (h* HostSlice) Cap() int {
	return cap(h.inner)
}

func (h* HostSlice) IsEmpty() bool {
	return len(h.inner) == 0
}

func (h* HostSlice) AsSlice() []any {
		return h.inner
}

func (h* HostSlice) AsPointer() *any {
	return &h.inner[0]
}

func (h HostSlice) IsOnDevice() bool {
	return false
}

func Malloc(size uint) (DeviceSlice, cuda_bindings.CudaErrorT) {
	if size == 0 {
		return DeviceSlice{}, cuda_bindings.CudaErrorMemoryAllocation
	}

	var p C.void
	dp := unsafe.Pointer(&p)
	err := cuda_bindings.CudaMalloc(&dp, size)

	return DeviceSlice{inner: dp, capacity: int(size), length: 0}, err
}

// TODO
func MallocAsync() {}

func (d* DeviceSlice) Free() cuda_bindings.CudaErrorT {
	return cuda_bindings.CudaFree(d.inner)
}

func (d* DeviceSlice) CopyToHost(dst HostSlice, size uint) HostSlice {
	dst_c := unsafe.Pointer(&dst.inner[0])
	cuda_bindings.CudaMemcpy(dst_c, d.inner, uint64(size), cuda_bindings.CudaMemcpyDeviceToHost)
	return dst
}

func (d* DeviceSlice) CopyToHostAsync(dst HostSlice, size uint, stream cuda_bindings.CudaStream) {
	dst_c := unsafe.Pointer(&dst.inner[0])
	cuda_bindings.CudaMemcpyAsync(dst_c, d.inner, uint64(size), cuda_bindings.CudaMemcpyDeviceToHost, stream)
	return
}

func (h* HostSlice) CopyFromHost(dst DeviceSlice, size uint) DeviceSlice {
	src_c := unsafe.Pointer(&h.inner[0])
	cuda_bindings.CudaMemcpy(dst.inner, src_c, uint64(size), cuda_bindings.CudaMemcpyHostToDevice)
	return dst
}

func (h* HostSlice) CopyFromHostAsync(dst DeviceSlice, size uint, stream cuda_bindings.CudaStream) {
	src_c := unsafe.Pointer(&h.inner[0])
	cuda_bindings.CudaMemcpyAsync(dst.inner, src_c, uint64(size), cuda_bindings.CudaMemcpyHostToDevice, stream)
	return
}
