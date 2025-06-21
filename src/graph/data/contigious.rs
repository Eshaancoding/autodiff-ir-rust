use crate::core::node::{Tensor, NodeTrait};
use crate::{ir_b_add, ir_b_id, Value, IRCmds};

#[derive(Clone)]
pub struct ContigiousNode {
    parent: Tensor,
    val: Option<Value>
}

impl NodeTrait for ContigiousNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val() {
            return v;
        }

        let p_val = self.parent.forward();  
        let res_val = c_contigious(&p_val, self.val.as_ref().map(|v| v.id.clone()));
        self.val = Some(res_val.clone());
        res_val 
    }        

    fn dim (&self) -> Vec<usize> {
        self.parent.dim()        
    }

    fn backward (&mut self, grad:crate::Value) {
        self.parent.n.borrow_mut().backward(grad);
    }

    fn val (&self) -> Option<Value> {
        if self.parent.val().is_none() { return None }
        self.val.clone()
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone()) 
    }
}

impl Tensor {
    pub fn contigious (&self) -> Tensor {
        Tensor::new(ContigiousNode {
            parent: self.clone(),
            val: None
        })
    }
}

fn c_contigious (a: &Value, id: Option<String>) -> Value {
    let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
    ir_b_add(IRCmds::Contigious { 
        a: a.id.clone(),
        res: id.clone()
    });

    Value {
        dim: a.dim.clone(),
        id
    }
}