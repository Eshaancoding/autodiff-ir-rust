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

    // val
    fn val (&self) -> Value;

    // === grad ===
    fn grad (&self) -> Value {
        panic!("Calling .grad() on a non-tensor node");
    }

    fn reset_grad (&mut self) {
        panic!("Calling .reset_grad() on a non-tensor node");
    }

    // Track whether constant node
    fn is_const (&self) -> bool { false }
}

// this is just a concrete type over the traits; so we can use operation overloading
#[derive(Clone)]
pub struct Tensor {
    // Yes, it's not the most safest implementation. Let me know if there's any other methods
    // I skimmed over df/dx, which doesn't use Rc<RefCell<>>. However, a refactor that big will take some time.
    pub n: Rc<RefCell<dyn NodeTrait>>,
}

impl Tensor {
    pub fn new<T: NodeTrait + 'static> (n: T) -> Tensor {
        Tensor {
            n: Rc::new(RefCell::new(n))
        }
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

    pub fn val (&self) -> Value {
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
        self.val().to_node()
    }

    pub fn detach_grad (&self) -> Tensor {
        self.forward(); 
        self.val().to_node_with_grad()
    }

    pub fn is_const (&self) -> bool {
        self.n.borrow().is_const()
    }
}