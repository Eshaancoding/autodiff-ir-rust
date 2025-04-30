use crate::{autodiff, Tensor, Module};

use super::SeqF;

pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>
}

impl Module for Linear {
    fn params (&self) -> Vec<Tensor> {
        if self.b.is_some() {
            vec![self.w.clone(), self.b.clone().unwrap()] 
        } else {
            vec![self.w.clone()] 
        }
    }
}

impl SeqF for Linear {
    fn f (&self, x: Tensor) -> Tensor {
        let mut res = autodiff::dot(x, self.w.clone());
        if let Some(b) = &self.b {
            res += b.clone();
        }
        res
    }
}

// my sneaky way of doing a "constructor" so to speak
// Rust somehow allows function names and struct Names to be the exact same thing
#[allow(non_snake_case)]
pub fn Linear (inp: usize, out: usize, bias: bool) -> Linear {
    Linear {
        w: autodiff::randn(vec![inp, out]),
        b: if bias {Some(autodiff::randn(vec![out]))} else {None},
    }
}

#[allow(non_snake_case)]
pub fn LinearWithWeights (w: Tensor, b: Option<Tensor>) -> Linear {
    if let Some(b_s) = b.clone() {
        assert_eq!(b_s.dim()[0], w.dim()[1], "weight dim [a,b] must match bias dim [b]");
    }

    Linear {
        w,
        b,
    }
}