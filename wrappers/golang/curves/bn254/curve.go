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

func (s* ScalarField) GenerateRandom(size int) []ScalarField {
	scalars := make([]ScalarField, size)
	scalarsP := unsafe.Pointer(&scalars[0])
	scalarsC := (*C.BN254_scalar_t)(scalarsP)
	C.GenerateScalars(scalarsC, size)
	
	return scalars
}

type Projective struct {
	core.Projective
}

type Affine struct {
	core.Affine
}
