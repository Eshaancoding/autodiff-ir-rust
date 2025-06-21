use crate::core::{node::NodeTrait, value::Value, Tensor};
use crate::ir::IRCmds;
use crate::{ir_b_add, ir_b_id};

// ============ Dot Node (operation) ============ 
#[derive(Clone)]
pub struct DotProductNode {
    left: Tensor,
    right: Tensor,
    val: Option<Value>
}

// ================== Basic Functionality ================== 
impl NodeTrait for DotProductNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val() {
            return v
        }

        let left = self.left.forward();
        let right = self.right.forward();
        let r_val = c_dot_product(&left, &right, self.val.as_ref().map(|v| v.id.clone()));
        self.val = Some(r_val.clone());
        r_val
    }

    fn dim (&self) -> Vec<usize> {
        vec![
            *self.left.dim().first().unwrap(),
            *self.right.dim().last().unwrap(),
        ]    
    }

    fn backward (&mut self, grad:Value) {
        let g_node = grad.to_node();
        self.left.n.borrow_mut().backward(
            dot(g_node.clone(), self.right.clone().t()).forward()
        );
        self.right.n.borrow_mut().backward(
            dot(self.left.clone().t(), g_node).forward()
        );
    }

    fn val (&self) -> Option<Value> {
        if self.left.val().is_none() { return None }
        if self.right.val().is_none() { return None }
        self.val.clone()
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone())
    }
}

// ============== Creating Dot Product Node ============ 
pub fn dot (a: Tensor, b: Tensor) -> Tensor { // wrapper is in dot product
    Tensor::new(DotProductNode {
        left: a,
        right: b,
        val: None
    })
}

// ============= Outer Product Core Funct ============= 
fn c_dot_product (left: &Value, right: &Value, id: Option<String>) -> Value {
    let a_dim = left.dim.clone();
    let b_dim = right.dim.clone();

    assert_eq!(a_dim.len(), 2, "A must be a 2d array"); 
    assert_eq!(b_dim.len(), 2, "B must be a 2d array"); 
    assert_eq!(a_dim[1], b_dim[0], "A's second dimension must equal B's first dimension");

    let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
    ir_b_add(IRCmds::DotProduct { 
        a: left.id.clone(),
        b: right.id.clone(),
        res: id.clone()
    });

    return Value {
        dim: vec![
            *left.dim.first().unwrap(),
            *right.dim.last().unwrap()
        ],
        id,
    }
}