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
        let res_val = c_view(&v, self.target_dim.clone(), self.val.as_ref().map(|v| v.id.clone()));
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        self.target_dim.clone()
    }

    fn backward (&mut self, grad:Value) {
        let g = c_view(
            &grad, 
            self.parent.dim().clone(),
            None
        );
        self.parent.n.borrow_mut().backward(g);
    }
    
    fn val (&self) -> Option<Value> {
        if self.parent.val().is_none() { return None }
        self.val.clone()
    }
    
    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone())
    }
}

// ==================== Helper functions for -1 @ view node ==================== 
pub fn handle_minus_dim (source_dim: &Vec<usize>, input_dim: &Vec<i32>) -> Vec<usize> {
    // check if -1
    if let Some(idx) = input_dim.iter().position(|x| *x == -1) {
        let total_size: usize = source_dim.iter().product();
        let inp_size: usize = -input_dim.iter().product::<i32>() as usize;
        
        let mut ret = input_dim.clone(); 
        ret[idx] = (total_size / inp_size) as i32;
        assert!(total_size % inp_size == 0, "Can't fill in -1");

        return ret.iter().map(|a| *a as usize).collect::<Vec<usize>>();
    }

    // if no -1, then just let this happen
    input_dim.iter().map(|a| *a as usize).collect::<Vec<usize>>()
}

// ================== Creating Node ================== 
impl Tensor {
    // you have to refactor the target dim such that it is i32 instead of usize
    pub fn view (&self, target_dim: Vec<i32>) -> Tensor {
        let source_dim = self.dim();
        let target_dim = handle_minus_dim(&source_dim, &target_dim);
        if target_dim != source_dim {
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
        let total_size = self.dim().iter().product::<usize>() as i32;
        Self::view(self, vec![total_size])
    }

    pub fn unsqueeze (&self, dim:i32) -> Tensor {
        let p_dim = self.dim().len();
        let dim = if dim < 0 { p_dim as i32 + dim + 1 } else { dim } as usize;

        let mut new_dim = self.dim().iter().map(|&i| i as i32).collect::<Vec<i32>>();
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
            Self::view(
                self, 
                new_dim.iter().map(|&i| i as i32).collect::<Vec<i32>>()
            )
        }
    }
}

// ================== Core Functionality ================== 
fn c_view (p: &Value, target_dim: Vec<usize>, id: Option<String>) -> Value {
    assert_eq!(
        target_dim.iter().product::<usize>(), 
        p.dim.iter().product::<usize>(), 
        "Target dim must be same size as input dim"
    );

    // add to IR
    let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
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