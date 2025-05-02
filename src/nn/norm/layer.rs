use crate::{autodiff, Tensor};
use crate::nn::{Module, SeqF};

pub struct LayerNorm {
    scale: Tensor,
    bias: Tensor,
}

impl Module for LayerNorm {
    fn params (&self) -> Vec<Tensor> {
        vec![self.scale.clone()]
    }
}

impl SeqF for LayerNorm {
    fn f (&self, x: Tensor) -> Tensor {
        let eps = 1e-9;

        let y = (x.clone() - x.mean(-1).unsqueeze(-1)) / 
                        (x.var(-1, 0) + eps).sqrt().unsqueeze(-1);
        y * self.scale.clone() + self.bias.clone()
    }
}

#[allow(non_snake_case)]
pub fn LayerNorm (d_model: usize) -> LayerNorm {
    LayerNorm { 
        scale: autodiff::ones(vec![d_model]),
        bias:  autodiff::zeros(vec![d_model]),
    }
}