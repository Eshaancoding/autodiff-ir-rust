// Attention mechanism
use crate::{autodiff, Tensor, Module};
use crate::nn::Linear;

// ======================= Attention ======================= 

// add masking
// change to bmm (impl first, using view and dot)
pub fn Attention (Q:Tensor, K:Tensor, V:Tensor, d_model:i32) -> Tensor {
    autodiff::dot(
        (
            autodiff::dot(Q, K.t()) / 
            (d_model as f64).sqrt() 
        ).softmax(-1),
        V
    )
}

// ======================= MultiHead Attention ======================= 

pub struct MultiHeadAttention {
    pub WQ: Linear,
    pub WK: Linear,
    pub WV: Linear,
    pub num_heads: u32, 
}

pub Module for MultiHeadAttention {
    fn params (&self) -> Vec<Tensor> { 
        vec![self.WQ.clone(), self.WK.clone(), self.WV.clone() ] 
    }
}

pub SeqF for MultiHeadAttention {
    fn f (&self, x: Tensor) -> Tensor {
        
    }
}

#[allow(non_snake_case)]
pub fn MultiHeadAttention (d_model: u32, num_heads: u32) -> MultiHeadAttention {
    
}

// ======================= MultiHead Attention ======================= 