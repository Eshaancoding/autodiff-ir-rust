// ============ Value Class (representing forward/backward propogation values) ============ 
// This the class actually representing the computation. 
// Node, Tensor, Ops, etc. are classes used to just build the computation graph. 

use std::sync::Arc;
use crate::{IRCmds, NodeTrait, ValueData};
use super::{add_to_dep, ir_b_add, ir_b_id, Tensor, TensorNode, DEVICE};

// Value just stores the dimension as we go through forward and backward propogation. However, the actual value is computed when we execute IR
#[derive(Clone, Debug)]
pub struct Value {
    pub dim: Vec<usize>,
    pub id: String
}

impl Value {
    // helper functions for creating value
    // These functions should also be for `Node`
    pub fn ones (dim: Vec<usize>) -> Value {
        Self::fill(1.0, dim) 
    }

    pub fn zeros (dim: Vec<usize>) -> Value {
        Self::fill(0.0, dim) 
    }

    pub fn empty () -> Value {
        Self {
            dim: vec![],
            id: "".to_string()
        }
    }

    pub fn fill (val: f64, dim: Vec<usize>) -> Value {
        Self::new( 
            vec![val; dim.iter().product()], 
            dim 
        )
    }

    pub fn val (val: f64) -> Value {
        Self::new(
            vec![val],
            vec![1]
        )        
    }

    // TensorNode is just a wrapper around Value, technically.
    // Will be a constant
    // For node, this is similar to .detach(), but gradient will exist normally
    // This is helpful for:
    // Gradient Value --> Node --> perform operations -(compile to)-> IR
    pub fn to_node (&self) -> Tensor {
        Tensor::new(ValueNode {
            v: self.clone()
        })
    }

    pub fn to_node_with_grad (&self) -> Tensor {
        Tensor::new(TensorNode {
            v: self.clone(),
            gd: None
        })
    }

    pub fn new (data: Vec<f64>, dim: Vec<usize>) -> Value {
        let id = ir_b_id(); 
        ir_b_add(IRCmds::CreateMat { 
            contents: Arc::new(data.clone()),
            dim: dim.clone(), 
            id: id.clone() 
        });
        
        

        // call IR
        Value {
            dim,
            id
        }
    }

    pub fn get (&self) -> ValueData {
        let mut guard = DEVICE.lock().unwrap();
        let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");

        let res = ir_b.get_tensor(&self.id);
        drop(guard);
        res
    }

    /*
    Ensures that variable will be kept when optimizing dependency list
    * Note that this is automatically done for the value and the gradient of a declared autodiff::tensor. 
    * However, for tensor::empty, there is a chance you might need to call this manually within your script.
    */
    pub fn keep (&self) {
        add_to_dep(self.id.clone());
    }
}

// Value node
// This is important for turning the gradient into a node to be manipulated by other variables
#[derive(Clone)]
pub struct ValueNode {
    pub v: Value
}

impl NodeTrait for ValueNode {
    fn forward (&mut self) -> Value {
        self.v.clone() 
    } 

    fn backward (&mut self, _:Value) {
        // no backwards for constant 
    }

    fn dim (&self) -> Vec<usize> {
        self.v.dim.clone()
    }

    fn val (&self) -> Option<Value> {
        Some(self.v.clone())
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone()) 
    }
}

