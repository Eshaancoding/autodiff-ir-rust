use crate::Tensor;
use crate::nn::{Module, SeqF};

pub struct Softmax {
    dim: i32
}

impl Module for Softmax {
    fn params (&self) -> Vec<Tensor> {
        vec![]
    }
}

impl SeqF for Softmax {
    fn f (&self, x: Tensor) -> Tensor {
        x.exp() / x.exp().sum(self.dim)
    }
}

#[allow(non_snake_case)]
pub fn Softmax (dim: i32) -> Softmax {
    Softmax {
        dim
    }
}

impl Tensor {
    pub fn softmax (&self, dim:i32) -> Tensor {
        self.exp() / self.exp().sum(dim)
    }
}