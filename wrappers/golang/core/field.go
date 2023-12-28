package core

import (
	"encoding/binary"
)

type Field struct {
	NumLimbs int8
	limbs []uint64 // need to constrain this by NumLimbs
}

func (f* Field) GetLimbs() []uint64 {
	return f.limbs
}

func (f* Field) FromLimbs(limbs []uint64) Field {
	f.NumLimbs = int8(len(limbs))
	f.limbs = limbs

	return *f
}

func (f* Field) Zero() Field {
	f.limbs = make([]uint64, f.NumLimbs)

	return *f
}

func (f* Field) One() Field {
	f.limbs = make([]uint64, f.NumLimbs)
	f.limbs[0] = 1

	return *f
}

func (f* Field) FromBytesLittleEndian(bytes []byte) Field {
	limbs := make([]uint64, f.NumLimbs)
	for i := int8(0); i < f.NumLimbs; i++ {
		limbs[i] = binary.LittleEndian.Uint64(bytes[i:i+8])
	}

	return *f
}

func (f* Field) ToBytesLittleEndian() []byte {
	bytes := make([]byte, f.NumLimbs*8)
	for i, v := range f.limbs {
		binary.LittleEndian.PutUint64(bytes[i*4:], v)
	}

	return bytes
}
