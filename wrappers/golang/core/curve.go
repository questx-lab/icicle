package core

// #cgo CFLAGS: -I./include/
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254
import "C"
import "unsafe"

/*****************************************************
******************************************************
************ 	Projective Implementation		************
******************************************************
*****************************************************/

type Projective struct {
	X, Y, Z Field
}

func (p* Projective) Zero() Projective {
	p.X.Zero()
	p.Y.Zero()
	p.Z.Zero()
	
	return *p
}

func (p* Projective) FromLimbs(x, y, z []uint64) Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p* Projective) FromAffine(a Affine) Projective {
	z := Field {
		NumLimbs: a.X.NumLimbs,
	}
	
	return Projective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}

// TODO: see if these should be moved to curve specific??
func (p* Projective) ToAffine() Affine {
	a := Affine {}
	C.ToAffine(unsafe.Pointer(p), unsafe.Pointer(&a))
	return a
}

func (p* Projective) Eq(p2* Projective) bool {
	return C.Eq(unsafe.Pointer(p), unsafe.Pointer(p2)) != 0
}

func (p* Projective) GenerateRandom(size int) []Projective {
	points := make([]Projective, size)
	pointsP := unsafe.Pointer(&points[0])
	pointsC := (*C.BN254_projective_t)(pointsP)
	C.GenerateProjectivePoints(pointsC, size)
	
	return points
}

/*****************************************************
******************************************************
************ 		Affine Implementation		**************
******************************************************
*****************************************************/ 

type Affine struct {
	X, Y Field
}

func (a* Affine) Zero() Affine {
	a.X.Zero()
	a.Y.Zero()
	
	return *a
}

func (a* Affine) FromLimbs(x, y []uint64) Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a* Affine) FromProjective(p Projective) Affine {
	return p.ToAffine()
}

func (a* Affine) ToProjective() Projective {
	z := Field {
		NumLimbs: a.X.NumLimbs,
	}
	
	return Projective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}

func (a* Affine) GenerateRandom(size int) []Affine {
	points := make([]Affine, size)
	pointsP := unsafe.Pointer(&points[0])
	pointsC := (*C.BN254_affine_t)(pointsP)
	C.GenerateAffineePoints(pointsC, size)
	
	return points
}
