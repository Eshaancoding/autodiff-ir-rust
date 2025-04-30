use crate::Tensor;
use crate::nn::{Module, SeqF};

pub struct Sigmoid {}

impl Module for Sigmoid {
    fn params (&self) -> Vec<Tensor> {
        vec![]
    }
}

impl SeqF for Sigmoid {
    fn f (&self, x: Tensor) -> Tensor {
        1.0 / (1.0 + x.neg().exp())    
    }
}

#[allow(non_snake_case)]
pub fn Sigmoid () -> Sigmoid {
    Sigmoid {}
}