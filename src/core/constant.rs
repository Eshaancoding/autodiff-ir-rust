use super::{NodeTrait, Value};

#[derive(Clone)]
pub struct ConstantNode {
    pub v: Value
}

impl NodeTrait for ConstantNode {
    fn forward (&mut self) -> Value {
        self.v.clone() 
    } 

    fn backward (&mut self, _:Value) {
        // no backwards for constant 
    }

    fn dim (&self) -> Vec<usize> {
        self.v.dim.clone()
    }

    fn val (&self) -> Value {
        self.v.clone()
    }

    fn is_const (&self) -> bool {
        true 
    }
} 