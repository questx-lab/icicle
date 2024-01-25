package core

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
