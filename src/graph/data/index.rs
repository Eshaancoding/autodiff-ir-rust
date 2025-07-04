use std::ops::Range;

use crate::{concat, ir_b_add, ir_b_id};
use crate::core::autodiff;
use crate::core::node::{Tensor, NodeTrait};
use crate::core::value::Value;
use crate::ir::IRCmds;

// ============ Select Node (operation) ============ 
// always assumed first dim; if you don't want that then you have to use permute()
#[derive(Clone)]
pub struct IndexNode { 
    parent: Tensor,
    idx: usize,
    dim: usize,
    val: Option<Value>,
}

// ================== Declaring NodeTraits (constructing graph) ================== 
impl NodeTrait for IndexNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val() { 
            return v;
        }
        let p_val = self.parent.forward();
        let res_val = c_idx(&p_val, self.idx, self.dim, self.val.as_ref().map(|i| i.id.clone()));
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        let mut d = self.parent.dim().clone();
        d.remove(self.dim);
        d
    }

    fn backward (&mut self, grad: Value) {
        let i_dim = self.parent.dim().clone();

        let mut zero_first_dim = i_dim.clone();
        zero_first_dim[self.dim] = self.idx;

        let mut zero_second_dim = i_dim.clone();
        zero_second_dim[self.dim] = i_dim[0] - self.idx - 1;

        let first_zero = Value::zeros(zero_first_dim).to_node();
        let second_zero = Value::zeros(zero_second_dim).to_node();
        let grad_n = grad.to_node();

        self.parent.n.borrow_mut().backward(
            concat(vec![first_zero, grad_n.unsqueeze(0), second_zero], self.dim as i32).forward()
        ); 
    }

    fn val (&self) -> Option<Value> {
        if self.parent.val().is_none() { return None } 
        self.val.clone()
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone()) 
    }
}

// ================== Creating Node ================== 
impl Tensor {
    // select index at dim
    pub fn i (&self, i: usize, dim: i32) -> Tensor {
        let p_dim = self.dim();
        let dim = if dim < 0 { p_dim.len() as i32 + dim } else { dim } as usize;

        assert!(dim < p_dim.len(), "invalid dimension!");
        assert!(i < p_dim[dim], "invalid index!");

        Tensor::new(IndexNode {
            parent: self.clone(),
            idx: i,
            dim,
            val: None
        })
    }

    // add more using permute & map
    pub fn r (&self, r: Range<usize>, dim: i32) -> Tensor {
        autodiff::concat(
            r.map(|i| self.i(i, dim).unsqueeze(dim)).collect::<Vec<Tensor>>(), 
            dim
        )
    }
}

// ============= Index Node Core Func ============
fn c_idx (a: &Value, idx: usize, dim: usize, id: Option<String>) -> Value {
    let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
    ir_b_add(IRCmds::Index { 
        a: a.id.clone(), 
        index: idx.clone(), 
        dim: dim.clone(),
        res: id.clone()
    });

    // find target dim
    let mut d = a.dim.clone(); 
    d.remove(dim);

    // Return value
    Value {
        dim: d,
        id
    }
}