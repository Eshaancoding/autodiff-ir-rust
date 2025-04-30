use crate::{core::{node::{NodeTrait, Tensor}, value::Value}, ir_b_add, ir_b_id};
use crate::ir::IRCmds;

// ============ Permute Node (operation) ============ 
#[derive(Clone)]
pub struct PermuteNode {
    parent: Tensor,
    permute: Vec<usize>,
    val: Option<Value>
}

// ================== Declaring Permute Node (constructing graph) ================== 
impl NodeTrait for PermuteNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() { 
            return v;
        }
        let p_val = self.parent.forward();
        let res_val = c_permute(&p_val, self.permute.clone());
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        let c_dim = self.parent.dim().clone();
        let mut dim = vec![0; c_dim.len()];
        for i in 0..c_dim.len() {
            dim[i] = c_dim[self.permute[i]];
        }
        return dim
    }

    fn backward (&mut self, grad:Value) {
        // calculate inverse permutation
        let mut inverse_permute = vec![0; self.permute.len()];
        for i in 0..self.permute.len() {
            inverse_permute[self.permute[i]] = i;
        }
        
        // then just permute grad and send
        self.parent.n.borrow_mut().backward(c_permute(&grad, inverse_permute));
    }

    fn val (&self) -> Value {
        self.val.clone().expect("Run forward before running val") 
    }
}

// ============= Creating Func ============
impl Tensor {
    pub fn permute (&self, p: &Vec<usize>) -> Tensor { 
        // if p follows a sequential pattern 0, 1, 2, etc. then just return the clone
                
        Tensor::new(PermuteNode {
            parent: self.clone(),
            permute: p.clone(),
            val: None
        })
    }

    pub fn t (&self) -> Tensor {
        assert_eq!(self.dim().len(), 2, "Transpose only supprots 2D matrix");
        Tensor::new(PermuteNode {
            parent: self.clone(),
            permute: vec![1, 0],
            val: None
        }) 
    }
}

// ============= Permute Node Core Func ============
fn c_permute (a: &Value, permute: Vec<usize>) -> Value {
    // find if dim mismatch
    assert_eq!(permute.len(), a.dim.len(), "Permute dimension mismatch");
    for i in permute.iter() {
        if *i >= a.dim.len() {
            panic!("Permute index out of bounds");
        }
    }

    // if sequential, no need to add any command
    let mut is_sequential = true;
    permute.iter().enumerate().for_each(|(i, v)| {
        if i != *v {
            is_sequential = false;
        }
    });
    if is_sequential { return a.clone(); }

    let id = ir_b_id();    
    ir_b_add(IRCmds::Permute { 
        a: a.id.clone(), 
        p: permute.clone(), 
        res: id.clone() 
    });

    // find dim
    let c_dim = a.dim.clone();
    let mut dim = vec![0; c_dim.len()];
    for i in 0..c_dim.len() {
        dim[i] = c_dim[permute[i]];
    }

    Value {
        dim,
        id
    }
}