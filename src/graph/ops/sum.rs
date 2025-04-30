use crate::{ir_b_add, ir_b_id, Tensor, NodeTrait, Value, IRCmds};

pub struct SumNode {
    parent: Tensor,
    dim: usize,
    val: Option<Value>
}

impl NodeTrait for SumNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() { 
            return v;
        }       

        let val = self.parent.forward();
        let res_val = c_sum(&val, self.dim);
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        let mut d = self.parent.dim().clone();
        d.remove(self.dim);
        d
    }

    fn backward (&mut self, grad:crate::Value) {
        let repeat_n = self.parent.dim()[self.dim];    
        self.parent.n.borrow_mut().backward(
            grad.to_node().repeat(
                |n, _| n.clone(), 
                repeat_n, 
                self.dim
            ).forward()
        ); 
    }

    fn val (&self) -> crate::Value {
        self.val.clone().expect("Need to run forward before val")    
    }
}

impl Tensor {
    pub fn sum (&self, dim:usize) -> Tensor {
        Tensor::new(SumNode {
            parent: self.clone(),
            dim,
            val: None
        })
    }
}

fn c_sum (a: &Value, dim: usize) -> Value {
    let id = ir_b_id();
    ir_b_add(IRCmds::Sum { 
        a: a.id.clone(), 
        dim,
        res: id.clone()
    });

    let mut d = a.dim.clone();
    d.remove(dim);

    Value { 
        dim: d, 
        id
    }
}