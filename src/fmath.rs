//! Deterministic transcendental functions.
//!
//! `f64::sin` & co. call the platform libm (glibc natively, the `libm` crate's
//! musl port on wasm32), which differ by an ulp here and there. Routing every
//! transcendental call through the `libm` crate makes the native CLI and the
//! wasm module produce bit-identical output on every platform.

pub trait DetMath {
    fn lsin(self) -> f64;
    fn lcos(self) -> f64;
    fn ltan(self) -> f64;
    fn latan(self) -> f64;
    fn latan2(self, x: f64) -> f64;
    fn lasin(self) -> f64;
    fn lacos(self) -> f64;
    fn lexp(self) -> f64;
    fn lln(self) -> f64;
    fn lpowf(self, y: f64) -> f64;
    fn lhypot(self, y: f64) -> f64;
}

impl DetMath for f64 {
    #[inline]
    fn lsin(self) -> f64 {
        libm::sin(self)
    }
    #[inline]
    fn lcos(self) -> f64 {
        libm::cos(self)
    }
    #[inline]
    fn ltan(self) -> f64 {
        libm::tan(self)
    }
    #[inline]
    fn latan(self) -> f64 {
        libm::atan(self)
    }
    #[inline]
    fn latan2(self, x: f64) -> f64 {
        libm::atan2(self, x)
    }
    #[inline]
    fn lasin(self) -> f64 {
        libm::asin(self)
    }
    #[inline]
    fn lacos(self) -> f64 {
        libm::acos(self)
    }
    #[inline]
    fn lexp(self) -> f64 {
        libm::exp(self)
    }
    #[inline]
    fn lln(self) -> f64 {
        libm::log(self)
    }
    #[inline]
    fn lpowf(self, y: f64) -> f64 {
        libm::pow(self, y)
    }
    #[inline]
    fn lhypot(self, y: f64) -> f64 {
        libm::hypot(self, y)
    }
}
