use crate::{autodiff, Tensor};
use crate::nn::{Module, SeqF};

pub struct RMS {
    scale: Tensor
}

impl Module for RMS {
    fn params (&self) -> Vec<Tensor> {
        vec![self.scale.clone()]
    }
}

impl SeqF for RMS {
    fn f (&self, x: Tensor) -> Tensor {
        let eps = 1e-9;
        let var = x.pow2().mean(-1).unsqueeze(-1) + eps;
        let input_norm = x / var.sqrt();
        self.scale.clone() * input_norm
    }
}

#[allow(non_snake_case)]
pub fn RMS (d_model: usize) -> RMS {
    RMS { 
        scale: autodiff::ones(vec![d_model])
    }
}