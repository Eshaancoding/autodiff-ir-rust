use crate::ir::IRCmds;
use crate::core::node::{Tensor, NodeTrait};
use crate::core::value::Value;
use crate::{ir_b_add, ir_b_id};

// ============ Clone Node (operation) ============ 
#[derive(Clone)]
pub struct ConcatNode {
    nodes: Vec<Tensor>,
    dim: usize,
    val: Option<Value>
}

// ================== Basic Functionality ================== 
impl NodeTrait for ConcatNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() { 
            return v;
        }       

        let mut values: Vec<Value> = vec![];
        for node in self.nodes.iter() {
            let n = node.forward();
            values.push(n);
        }

        let res = c_concat(values, self.dim);
        self.val = Some(res.clone());
        res
    }

    fn dim (&self) -> Vec<usize> {
        let mut d = self.nodes[0].dim();
        for i in 1..self.nodes.len() {
            d[self.dim] += self.nodes[i].dim()[self.dim];
        }

        d
    }

    fn backward (&mut self, grad:Value) {
        let mut c: usize = 0;
        for i in 0..self.nodes.len() {
            let d_interest = self.nodes[i].dim()[self.dim];
            let g = grad.to_node().r(c..(c+d_interest), self.dim as i32).forward();
            self.nodes[i].n.borrow_mut().backward(g);
            c += d_interest;
        } 
    }

    fn val (&self) -> Value {
        self.val.clone().expect("Run forward before getting value")
    }
}

// =================== Creating Node =================== 
pub fn concat (nodes: Vec<Tensor>, dim: i32) -> Tensor { // light wrapper in autodiff.rs
    let n_dim = nodes.first().expect("Empty concat tensor").dim().len();
    let dim = if dim < 0 { n_dim as i32 + dim } else { dim } as usize; // handles negative dim
    Tensor::new(ConcatNode {
        nodes,
        dim,
        val: None
    })
}

// ============= Concat Core Functionality ============
fn c_concat (nodes: Vec<Value>, dim: usize) -> Value { // main function used here
    // check dimension
    let mut first_dim: Vec<usize> = vec![];
    for i in nodes.iter() {
        if first_dim.len() == 0 {
            first_dim = i.dim.clone();
        }
        else {
            let mut f_d = first_dim.clone();
            let mut i_d = i.dim.clone();
            assert_eq!(f_d.len(), i_d.len(), "Dimensions are not equal at concat");
            f_d.remove(dim);
            i_d.remove(dim);
            assert_eq!(f_d, i_d, "Dim misaligned at concat");
        }
    }    

    // add to IR
    // continously apply concat
    let mut prev_node_id = nodes[0].id.clone();
    let mut total_d = nodes[0].dim.clone();
    for i in 1..nodes.len() {
        let node_id = nodes[i].id.clone();
        let new_id = ir_b_id();

        ir_b_add(IRCmds::Concat {
            a: prev_node_id,
            b: node_id,
            dim: dim.clone(),
            res: new_id.clone()
        });

        prev_node_id = new_id;
        total_d[dim] += nodes[i].dim[dim];
    }

    // Add value
    Value { dim: total_d, id: prev_node_id }
}