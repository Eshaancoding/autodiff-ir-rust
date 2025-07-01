use crate::Tensor;

pub struct SGD {
    n: Vec<Tensor>,
    lr: f32
}

impl SGD {
    pub fn step (&mut self) {
        for p in self.n.iter_mut() {
            *p -= self.lr * p.grad().to_node();
            p.forward();
        }
    }

    pub fn zero_grad (&mut self) {
        for p in self.n.iter_mut() {
            p.reset_grad();
        }
    }
}

#[allow(non_snake_case)]
pub fn SGD (m: Vec<Tensor>, lr: f32) -> SGD {
    SGD {
        n: m,
        lr
    }
}