use crate::{SeqF, Tensor};

use super::Module;

pub struct Sequential {
    n: Vec<Box<dyn SeqF>>
}

impl Sequential {
    pub fn insert (&mut self, func: impl SeqF + 'static) {
        self.n.push(Box::new(func)); 
    }
}

impl Module for Sequential {
    fn params (&self) -> Vec<Tensor> {
        let mut x: Vec<Tensor> = vec![];
        
        for i in self.n.iter() {
            x.extend(i.params().iter().map(|v| v.clone()));
        }

        x
    }
}

impl SeqF for Sequential {
    fn f (&self, x: Tensor) -> Tensor {
        let mut res = x.clone();
        for layer in self.n.iter() {
            res = layer.f(res)
        }

        res
    }
}

#[allow(non_snake_case)]
pub fn Sequential () -> Sequential {
    Sequential { n: vec![] }
}