use crate::{ir_b_add, ir_b_id, IRCmds};

use super::{NodeTrait, Value};

#[derive(Clone)]
pub struct ConstantNode {
    content: f64, 
    val: Option<Value>
}

impl ConstantNode {
    pub fn new (content: f64) -> ConstantNode {
        ConstantNode { content, val: None }
    }
}

impl NodeTrait for ConstantNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() {
            v
        }  else {
            let v = c_constant(self.content);
            self.val = Some(v.clone());
            v
        }
    } 

    fn backward (&mut self, _:Value) {
        // no backwards for constant 
    }

    fn dim (&self) -> Vec<usize> {
        vec![1]
    }

    fn val (&self) -> Value {
        self.val.clone().expect("Run forward propogation before running .val()")
    }

    fn is_const (&self) -> bool {
        true 
    }
} 

fn c_constant (contents: f64) -> Value {
    let id = ir_b_id();
    ir_b_add(IRCmds::CreateConstant { contents, id: id.clone() });

    return Value {
        dim: vec![1],
        id
    }
}