use crate::ir::IRCmds;
use crate::core::node::{Tensor, NodeTrait};
use crate::core::value::Value;
use crate::{ir_b_add, ir_b_id};

// ============ View Node ============ 
#[derive(Clone)]
pub struct ViewNode {
    parent: Tensor,
    target_dim: Vec<usize>,
    val: Option<Value>
}

// ================== Basic Functionality ================== 
impl NodeTrait for ViewNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() { 
            return v;
        }
        
        let v = self.parent.forward();
        let res_val = c_view(&v, self.target_dim.clone());
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        self.target_dim.clone()
    }

    fn backward (&mut self, grad:Value) {
        let g = c_view(
            &grad, 
            self.parent.dim().clone()
        );
        self.parent.n.borrow_mut().backward(g);
    }
    
    fn val (&self) -> Value {
        self.val.clone().expect("Need to run forward pass before val")
    }
}

// ================== Creating Node ================== 
impl Tensor {
    pub fn view (&self, target_dim: Vec<usize>) -> Tensor {
        if target_dim != self.dim() {
            Tensor::new(ViewNode {
                parent: self.clone(),
                target_dim,
                val: None
            })
        } else {
            self.clone()
        }
    }

    pub fn flatten (&self) -> Tensor {
        let total_size = self.dim().iter().product();
        Self::view(self, vec![total_size])
    }

    pub fn unsqueeze (&self, dim:usize) -> Tensor {
        let mut new_dim = self.dim().clone();
        new_dim.insert(dim, 1);
        Self::view(self, new_dim)
    }

    pub fn squeeze (&self, dim:usize) -> Tensor {
        if self.dim().len() <= 1 {
            self.clone()
        }
        else {
            let mut new_dim = self.dim().clone();
            assert_eq!(new_dim[dim], 1, "Squeeze dim must be one");
            new_dim.remove(dim);
            Self::view(self, new_dim)
        }
    }
}

// ================== Core Functionality ================== 
fn c_view (p: &Value, target_dim: Vec<usize>) -> Value {
    assert_eq!(
        target_dim.iter().product::<usize>(), 
        p.dim.iter().product::<usize>(), 
        "Target dim must be same size as input dim"
    );

    // add to IR
    let id = ir_b_id();
    ir_b_add(IRCmds::View { 
        a: p.id.clone(), 
        target_dim: target_dim.clone(), 
        res: id.clone() 
    });
    
    Value {
        dim: target_dim,
        id
    }
}