use crate::Value;
use std::rc::Rc;
use std::cell::RefCell;

// ============ General Node Class (constructing computation graph) ============ 
pub trait NodeTrait {
    // Forward propogation; updates val
    fn forward (&mut self) -> Value;

    // Backward Propogation; updates grad
    fn backward (&mut self, grad:Value);

    // For dimensional checking and needs shape for Grad operations
    fn dim (&self) -> Vec<usize>;  

    // gets the value of the node. However, could be None if the value is None (not forward) OR if any of the child nodes value is None (has to be recomputed)
    fn val (&self) -> Option<Value>;

    // === grad ===
    fn grad (&self) -> Value {
        panic!("Calling .grad() on a non-tensor node");
    }

    fn reset_grad (&mut self) {
        panic!("Calling .reset_grad() on a non-tensor node");
    }

    // Track whether constant node
    fn is_const (&self) -> bool { false }

    // If we need to deep copy the node, it calls this function
    // What usually happens when you do .clone() on a Tensor is that it just increments the reference.
    // However, this actually clones the entire node and returns and new memory location RefCell
    fn deep_copy (&self) -> Box<dyn NodeTrait>;
}

// this is just a concrete type over the traits; so we can use operation overloading
#[derive(Clone)]
pub struct Tensor {
    // Yes, it's not the most safest implementation. Let me know if there's any other methods
    // I skimmed over df/dx, which doesn't use Rc<RefCell<>>. However, a refactor that big will take some time.
    pub n: Rc<RefCell<Box<dyn NodeTrait>>>,
}

impl Tensor {
    pub fn new<T: NodeTrait + 'static> (n: T) -> Tensor {
        Tensor {
            n: Rc::new(RefCell::new(Box::new(n)))
        }
    }

    pub fn replace<T: NodeTrait + 'static> (&self, node: T) {
        let _ = self.n.replace(Box::new(node));
    }

    // implement NodeTraits
    pub fn forward (&self) -> Value {
        let result_value = self.n.borrow_mut().forward();
        result_value
    }

    pub fn backward (&self) {
        let dim = self.dim();         

        // start autograd with grad 1 (df/df)
        self.n.borrow_mut().backward(Value::ones(dim));
    }

    pub fn dim (&self) -> Vec<usize> {
        self.n.borrow().dim()
    }

    pub fn val (&self) -> Option<Value> {
        self.n.borrow().val()
    }

    pub fn grad (&self) -> Value {
        self.n.borrow().grad()
    }

    pub fn reset_grad (&self) {
        self.n.borrow_mut().reset_grad();
    }

    pub fn detach (&self) -> Tensor {
        self.forward(); 
        self.val().expect("Forward process went wrong").to_node()
    }

    pub fn detach_grad (&self) -> Tensor {
        self.forward(); 
        self.val().expect("Forward process went wrong").to_node_with_grad()
    }

    pub fn is_const (&self) -> bool {
        self.n.borrow().is_const()
    }

    pub fn deep_copy (&self) -> Tensor {
        Tensor {
            n: Rc::new(RefCell::new(self.n.borrow().deep_copy()))
        }
    }
}