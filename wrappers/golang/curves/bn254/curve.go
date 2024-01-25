package bn254

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
// #include "field.h"
import "C"
import (
	core "local/hello/icicle/wrappers/golang/core"
	"unsafe"
)

type ScalarField struct {
	core.Field
}

type BaseField struct {
	core.Field
}

const (
	NUM_SCALAR_LIMBS int8 = 4
	NUM_BASE_LIMBS int8 = 4
)

var bn254ScalarField = ScalarField {
	core.Field{
		NumLimbs: NUM_SCALAR_LIMBS,
	},
}

var bn254BaseField = BaseField {
	core.Field{
		NumLimbs: NUM_BASE_LIMBS,
	},
}

func GenerateScalars(size int) []ScalarField {
	scalars := make([]ScalarField, size)
	scalarsP := unsafe.Pointer(&scalars[0])
	scalarsC := (*C.BN254_scalar_t)(scalarsP)
	C.BN254GenerateScalars(scalarsC, size)
	
	return scalars
}

func convertScalarsMontgomery(values unsafe.Pointer, isInto bool) CudaError {
	C.BN254ScalarConvertMontgomery() // templatize this per curve??
}

func ToMontgomery(values unsafe.Pointer) CudaError {
	return convertScalarsMontgomery(values, true)
}

func FromMontgomery(values unsafe.Pointer) CudaError {
	return convertScalarsMontgomery(values, false)
}

type Projective struct {
	core.Projective
}

func GenerateProjectivePoints(size int, ) []Projective {
	points := make([]Projective, size)
	pointsP := unsafe.Pointer(&points[0])
	pointsC := (*C.BN254_projective_t)(pointsP)
	C.BN254GenerateProjectivePoints(pointsC, size)
	
	return points
}


func (p* Projective) ToAffine() Affine {
	a := Affine {}
	C.BN254ToAffine(unsafe.Pointer(p), unsafe.Pointer(&a))
	return a
}

func (p* Projective) Eq(p2* Projective) bool {
	return C.BN254Eq(unsafe.Pointer(p), unsafe.Pointer(p2)) != 0
}

type Affine struct {
	core.Affine
}

func (a* Affine) FromProjective(p Projective) Affine {
	return p.ToAffine()
}

func GenerateAffinePoints(size int) []Affine {
	points := make([]Affine, size)
	pointsP := unsafe.Pointer(&points[0])
	pointsC := (*C.BN254_affine_t)(pointsP)
	C.BN254GenerateAffinePoints(pointsC, size)
	
	return points
}
