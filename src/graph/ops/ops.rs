use crate::graph::data::broadcast::try_broadcast;
use crate::ir::IRCmds;
use crate::core::node::{Tensor, NodeTrait};
use crate::core::value::Value;
use crate::{autodiff, ir_b_add, ir_b_id};

macro_rules! create_op {
    (
        $node_name:ident,       // The node name
        $core_name:ident,       // the core function name (Value $op Value --> Value)
        $core_name_op_eql:ident,// the core name function for *=, +=, /=
        $ir_name:ident,         // IR name for multiplying two mat from IRCmds (Node / Node)
        $ir_name_op_eql:ident,  // IR name for *=, +=, /=
        $op:ident,              // the operation name under std::ops
        $op_func:ident,         // the func name under std::ops::$op
        $op_eq:ident,           // the operation name under std::ops of +=
        $op_eq_func:ident,      // the fun name under std::ops::$op_eq
        $l_bckwd:expr,          // expression for left (top) backward
        $r_bckwd:expr           // expression for right (bottom) backward
    ) => {
        // ============ Node (operation) ============ 
        #[derive(Clone)]
        pub struct $node_name {
            left: Tensor,
            right: Tensor,
            val: Option<Value>,
            is_op_equal: bool
        }

        // ================== Basic Functionality ================== 
        impl NodeTrait for $node_name {
            fn forward (&mut self) -> Value {
                if let Some(v) = self.val() { 
                    return v;
                } 

                let left = self.left.forward();
                let right = self.right.forward();

                let res_val = if self.is_op_equal { 
                    $core_name_op_eql(&left, &right) 
                } 
                else { 
                    let id = self.val.as_ref().map(|v| v.id.clone());
                    $core_name(&left, &right, id) 
                };

                self.val = Some(res_val.clone());
                res_val
            }

            fn dim (&self) -> Vec<usize> {
                if  self.left.is_const() { self.right.dim() } else { self.left.dim() }
                // broadcasting handles diff dimensions --> wraps to node
                // we just need to make sure it handles constant dimensioning
            }

            fn backward (&mut self, grad: Value) {
                let l = self.left.val().expect("l forward");
                let r = self.right.val().expect("r forward");
                
                self.left.n.borrow_mut().backward($l_bckwd(&l, &r, &grad));
                self.right.n.borrow_mut().backward($r_bckwd(&l, &r, &grad));    
            }
            
            fn val (&self) -> Option<Value> {
                if self.left.val().is_none() { return None }
                if self.right.val().is_none() { return None }
                self.val.clone()
            }

            fn grad (&self) -> Value {
                if self.is_op_equal {
                    self.left.grad()
                } else {
                    panic!("Calling .grad() on a non-tensor node");
                }
            }

            fn reset_grad (&mut self) {
                if self.is_op_equal {
                    self.left.reset_grad();
                } else {
                    panic!("Calling .reset_grad() on a non-tensor node")
                }
            }

            fn deep_copy (&self) -> Box<dyn NodeTrait> {
                Box::new(self.clone())
            }
        }

        // ============= Core Func ============
        fn $core_name (a: &Value, b: &Value, id: Option<String>) -> Value {
            let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
            
            // Assuming this gets called anyways from try_broadcasting
            // Core functions shouldn't have any checks; should be done at creation of the node itself
            // assert_eq!(a.dim, b.dim, "Dimensional mismatch +, -, *, / operation");

            ir_b_add(IRCmds::$ir_name {
                a: a.id.clone(),
                b: b.id.clone(),
                res: id.clone()
            });

            Value {
                dim: a.dim.clone(),
                id
            }
        }

        fn $core_name_op_eql (a: &Value, b: &Value) -> Value {
            // Assuming this gets called anyways from try_broadcasting
            // Core functions shouldn't have any checks; should be done at creation of the node itself
            // assert_eq!(a.dim, b.dim, "Dimensional mismatch +, -, *, / operation");

            ir_b_add(IRCmds::$ir_name_op_eql {
                s: a.id.clone(),
                o: b.id.clone()
            });

            Value {
                dim: a.dim.clone(),
                id: a.id.clone()
            }
        }

        // ================== Creating Node ================= 
        // Node + Node
        impl std::ops::$op<Tensor> for Tensor {
            type Output = Tensor;

            fn $op_func (self, other:Tensor) -> Tensor {
                let (a, b) = try_broadcast(&self, &other);

                Tensor::new($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: false
                })
            }
        }

        // Node + f32 constant
        impl std::ops::$op<f32> for Tensor {
            type Output = Tensor;

            fn $op_func (self, other:f32) -> Tensor {
                let (a, b) = try_broadcast(&self, &autodiff::constant(other, self.dim()));

                Tensor::new($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: false
                })
            }
        }

        // f32 constant + Node 
        impl std::ops::$op<Tensor> for f32 {
            type Output = Tensor;

            fn $op_func (self, other:Tensor) -> Tensor {
                let (a, b) = try_broadcast(&autodiff::constant(self, other.dim()), &other);

                Tensor::new($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: false
                })
            }
        }

        // ================== Creating Op Equal ================= 
        impl std::ops::$op_eq<Tensor> for Tensor {
            fn $op_eq_func (&mut self, rhs: Tensor) {
                let s_clone = self.deep_copy(); // since we are replacing the Rc<RefCell<>>, we need a deep copy of self before we destroy it
                let (a, b) = try_broadcast(&s_clone, &rhs);
                self.replace($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: true
                });
            }
        }

        impl std::ops::$op_eq<f32> for Tensor {
            fn $op_eq_func (&mut self, rhs: f32) {
                let s_clone = self.deep_copy(); // since we are replacing the Rc<RefCell<>>, we need a deep copy of self before we destroy it
                let (a, b) = try_broadcast(&s_clone, &autodiff::constant(rhs, self.dim()));

                self.replace($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: true
                });
            }
        }

        impl std::ops::$op_eq<Value> for Tensor {
            fn $op_eq_func (&mut self, rhs: Value) {
                let s_clone = self.deep_copy(); // since we are replacing the Rc<RefCell<>>, we need a deep copy of self before we destroy it
                let (a, b) = try_broadcast(&s_clone, &rhs.to_node());
                self.replace($node_name {
                    left: a,
                    right: b,
                    val: None,
                    is_op_equal: true
                });
            }
        }
    }
}

create_op!(
    AddNode, 
    c_add,     // core func name
    c_add_eql, // 
    ElwAdd,  // IR
    ElwAddEq,
    Add,     // std ops define
    add, 
    AddAssign,
    add_assign,
    |_: &Value, _: &Value, grad: &Value| -> Value { grad.clone() },  // left backward def
    |_: &Value, _: &Value, grad: &Value| -> Value { grad.clone() }   // right backward def
);

create_op!(
    MultiplyNode, 
    c_mult, 
    c_mult_eql,
    ElwMultiply, 
    ElwMultiplyEq,
    Mul, 
    mul, 
    MulAssign,
    mul_assign,
    |_: &Value, r: &Value, grad: &Value| -> Value { (r.to_node() * grad.to_node()).forward() }, // left
    |l: &Value, _: &Value, grad: &Value| -> Value { (l.to_node() * grad.to_node()).forward() }  // right
);

// ============= Negative ============= 
impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        -1.0 * self 
    }
}

// ============= Subtract ============= 
impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub (self, other:Tensor) -> Tensor {
        self + -1.0 * other
    }
}

impl std::ops::Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        self + -1.0 * other
    }
}

impl std::ops::Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub (self, other:f32) -> Tensor {
        self + (-1.0) * other 
    }
}

impl std::ops::SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        *self += -1.0 * rhs;
    }
}

impl std::ops::SubAssign<f32> for Tensor {
    fn sub_assign(&mut self, rhs: f32) {
        *self += rhs;
    }
}

impl std::ops::SubAssign<Value> for Tensor {
    fn sub_assign(&mut self, rhs: Value) {
        *self += -1.0 * rhs.to_node();
    }
}

// ============= Division ============= 
impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div (self, other:Tensor) -> Tensor {
        self * other.recip()
    }
}

impl std::ops::Div<Tensor> for f32 {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Self::Output {
        self * other.recip()
    }
}

impl std::ops::Div<f32> for Tensor {
    type Output = Tensor;

    fn div (self, other:f32) -> Tensor {
        self * (1.0/other)
    }
}

impl std::ops::DivAssign<Tensor> for Tensor {
    fn div_assign(&mut self, rhs: Tensor) {
        *self *= rhs.recip();
    }
}

impl std::ops::DivAssign<f32> for Tensor {
    fn div_assign(&mut self, rhs: f32) {
        *self *= 1.0 / rhs;
    }
}

impl std::ops::DivAssign<Value> for Tensor {
    fn div_assign(&mut self, rhs: Value) {
        *self *= rhs.to_node().recip();
    }
}