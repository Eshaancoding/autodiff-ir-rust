// Elementwise function: sin, cos, exp, etc.

use std::f64::consts::{FRAC_PI_2, E};
use crate::{IRCmds, Tensor, NodeTrait, Value, ir_b_id, ir_b_add};

macro_rules! create_func {
    ($n:ident, $core_func:ident, $st:ident, $ir:ident, $bckw:expr) => {
        // ============ Node (operation) ============ 
        #[derive(Clone)]
        pub struct $st {
            parent: Tensor,
            val: Option<Value>
        }

        // ============ Basic Functionality ============ 
        impl NodeTrait for $st {
            fn forward (&mut self) -> Value {
                if let Some(v) = self.val.clone() { 
                    return v;
                }

                let res_val = $core_func(&self.parent.forward()); 
                self.val = Some(res_val.clone());
                res_val
            }

            fn dim (&self) -> Vec<usize> {
                self.parent.dim()
            }          

            fn backward (&mut self, grad: Value) {
                let v = self.parent.val();
                self.parent.n.borrow_mut().backward($bckw(v, grad));
            }

            fn val (&self) -> Value {
                self.val.clone().expect("Run forward pass before calling .val()")
            }
        }

        impl Tensor {
            pub fn $n (&self) -> Tensor {
                Tensor::new($st {
                    parent: self.clone(),
                    val: None
                })
            }
        }

        // ============= Add ELW Core Func --> Value + Value ============
        pub fn $core_func (a: &Value) -> Value {
            let id = ir_b_id();
        
            ir_b_add(IRCmds::$ir {
                a: a.id.clone(),
                res: id.clone()
            });

            Value {
                dim: a.dim.clone(),
                id
            } 
        }
    };
}
    
create_func!(neg, c_neg, NegNode, Neg, |_, grad: Value| -> Value {
    // d/dx -x = -1
    (grad.to_node().neg()).forward()
}); 

create_func!(exp2, c_exp, ExpTwoNode, Exp2, |v: Value, grad: Value| -> Value {
    // d/dx 2^x = ln(2) * 2^x
    let mult_const = 2.0f64.ln();
    (grad.to_node() * mult_const * v.to_node().exp2()).forward()
});

create_func!(log2, c_logtwo, LogTwoNode, Log2, |v: Value, grad: Value| -> Value {
    // d/dx log_2(x) = log_2(e) * 1/x
    let con = E.log2();
    (grad.to_node() * v.to_node().recip() * con).forward()
});

create_func!(recip, c_recip, RecipNode, Recip, |v: Value, grad: Value| -> Value {
    // d/dx 1/x = -1 / (x^2)
    (grad.to_node() * (-1.0 / v.to_node().pow2())).forward()
});

create_func!(sqrt, c_sqrt, SqrtNode, Sqrt, |v: Value, grad: Value| -> Value {
    // d/dx sqrt(x) = d/dx x^(1/2) = 1/2 * x^(-1/2)
    (grad.to_node() * (1.0 / (2.0 * v.to_node().sqrt()))).forward()
});

create_func!(sin, c_sin, SinNode, Sin, |v: Value, grad: Value| -> Value {
    // d/dx sin(x) = cos(x)
    (grad.to_node() * v.to_node().cos()).forward()
});

impl Tensor {
    pub fn cos (&self) -> Tensor {
        (FRAC_PI_2 - self.clone()).sin()
    }

    pub fn exp (&self) -> Tensor {
        let mult_const = 1.0 / (2.0f64.ln());
        (self.clone() * mult_const).exp2()
    }

    pub fn ln (&self) -> Tensor {
        let mult_const = 2.0f64.ln();
        self.log2() * mult_const
    }

    pub fn pow2 (&self) -> Tensor {
        // rely on multiplication for pow2 and pow3
        self.clone() * self.clone()
    }

    pub fn pow3 (&self) -> Tensor {
        self.pow2() * self.clone()
    }

    pub fn pow (&self, p: Tensor) -> Tensor {
        // for general pow, use exp2 and log2 rule
        // x^p = exp2(p * log2(x))
        (p * self.log2()).exp2()
    }

    pub fn powf (&self, p: f64) -> Tensor {
        if p == 2.0 {
            self.pow2()
        } 
        else if p == 3.0 {
            self.pow3()
        }
        else {
            (p * self.log2()).exp2()
        }
    }
}