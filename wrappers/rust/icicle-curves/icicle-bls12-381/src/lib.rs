pub mod curve;
pub mod msm;
pub mod ntt;
pub mod poseidon;

impl icicle_core::SNARKCurve for curve::CurveCfg {}
