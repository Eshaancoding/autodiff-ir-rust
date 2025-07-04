use crate::{ir_b_add, ir_b_id, IRCmds};

use super::{NodeTrait, Value};

#[derive(Clone)]
pub struct ConstantNode {
    content: f32,
    dim: Vec<usize>, // need a dim to "expand" to. This is just to satisfy the dimension tracker
    val: Option<Value>
}

impl ConstantNode {
    pub fn new (content: f32, dim: Vec<usize>) -> ConstantNode {
        ConstantNode { content, val: None, dim }
    }
}

impl NodeTrait for ConstantNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() {
            v
        }  else {
            let v = c_constant(self.content, &self.dim);
            self.val = Some(v.clone());
            v
        }
    } 

    fn backward (&mut self, _:Value) {
        // no backwards for constant 
    }

    fn dim (&self) -> Vec<usize> {
        self.dim.clone()
    }

    fn val (&self) -> Option<Value> {
        self.val.clone()
    }

    fn is_const (&self) -> bool {
        true 
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone()) 
    }
} 

fn c_constant (contents: f32, dim: &Vec<usize>) -> Value {
    let id = ir_b_id();
    ir_b_add(IRCmds::CreateConstant { contents, id: id.clone(), dim: dim.clone() });

    return Value {
        dim: dim.clone(),
        id
    }
}