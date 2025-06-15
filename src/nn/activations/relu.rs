use crate::{autodiff, Tensor};
use crate::nn::{Module, SeqF};

pub struct ReLU {}

impl Module for ReLU {
    fn params (&self) -> Vec<Tensor> {
        vec![]
    }
}

impl SeqF for ReLU {
    fn f (&self, x: Tensor) -> Tensor {
        x.clone() * x.more_than(&autodiff::constant(0.0, x.dim()))
    }
}

#[allow(non_snake_case)]
pub fn ReLU () -> ReLU {
    ReLU {}
}

impl Tensor {
    pub fn relu (&self) -> Tensor {
        self.clone() * self.more_than(&autodiff::constant(0.0, self.dim()))
    }
}