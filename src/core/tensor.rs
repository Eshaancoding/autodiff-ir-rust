use crate::NodeTrait;
use crate::core::value::Value;

use super::is_harsh;

// ============ Tensor Node (representing input) ============ 
#[derive(Clone)]
pub struct TensorNode {
    pub v: Value,
    pub gd: Option<Value> // Grad value
}

impl NodeTrait for TensorNode {
    fn forward (&mut self) -> Value {
        /*
        We clone it because Value is where we are DONE with the computation graph, and wish to perform actual computation. 
        This is where we clone the data send it to the forward propogation
         */
        self.v.clone()
    }

    fn backward (&mut self, grad:Value) {
        println!("Calling backward on tensor v id: {} gd id: {}", self.v.id, grad.id);
        if let Some(gd_v) = self.gd.clone() {
            let mut gd_v_node = gd_v.to_node();
            gd_v_node += grad.to_node();
            gd_v_node.forward();
        } else {
            self.gd = Some(grad.clone());

            if !is_harsh() {
                self.gd.as_ref().unwrap().keep(); // add the gradient to dependency list
            }
        }
    }

    fn dim (&self) -> Vec<usize> {
        self.v.dim.clone()
    }

    fn val (&self) -> Option<Value> {
        Some(self.v.clone())
    }

    fn grad (&self) -> Value {
        self.gd.clone().expect("Must run backwards before calling grad()")
    }

    fn reset_grad (&mut self) {
        self.gd = None;
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone())  
    }
}

// whenever you update this; make sure you update
// =================== ADD EQUAL ==============
// acts as tensor