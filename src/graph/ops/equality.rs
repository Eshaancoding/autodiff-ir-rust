use crate::{autodiff, ir_b_add, ir_b_id, IRCmds, NodeTrait, Tensor, Value};

macro_rules! create_equality {
    (
        $node_name:ident,
        $core_name:ident,
        $ir_name:ident,
        $impl_name:ident
    ) => {
        #[derive(Clone)]
        pub struct $node_name {
            parent: Tensor,
            val: Option<Value> 
        }
        
        impl NodeTrait for $node_name {
            fn forward (&mut self) -> Value {
                if let Some(v) = self.val.clone() {
                    return v;
                }
                
                let val = self.parent.forward(); 
                let res_val = $core_name(&val);
                self.val = Some(res_val.clone());
                res_val 
            } 
            
            fn dim (&self) -> Vec<usize> {
                self.parent.dim() 
            }
            
            fn backward (&mut self, _:Value) {
                
                // Note: no grad on equality; use val * (equality statement) to create grad
                // However, we still set backward & grad because the resultant of equalities are treated as constants
            }

            fn grad (&self) -> Value {
                Value::zeros(self.dim()) 
            }
            
            fn val (&self) -> Value {
                self.val.clone().expect("Need to run forward pass before val")
            }
        }
        
        fn $core_name (a: &Value) -> Value {
            let id = ir_b_id();
            ir_b_add(IRCmds::$ir_name {
                a: a.id.clone(),
                res: id.clone()
            }); 
            
            Value {
                dim: a.dim.clone(),
                id
            }
        }

        impl Tensor {
            pub fn $impl_name (&self, b: &Tensor) -> Tensor {
                Tensor::new($node_name {
                    parent: self.clone() - b.clone(),
                    val: None
                })
            }
        }
    
    };
}

create_equality!(EqualityNode, c_equality, EqualZero, equal);
create_equality!(MoreZeroNode, c_more_zero, MoreZero, more_than);
create_equality!(LessZeroNode, c_less_zero, LessZero, less_than);

impl Tensor {
    // TODO: TEST!!!  
    pub fn all (&self) -> Tensor {
        let mut x = self.clone();
        let total_val = self.dim().iter().product::<usize>();
        for _ in x.dim() {
            x = x.sum(0);
        }

        x.equal(&autodiff::const_val(total_val as f64, vec![1])) 
    }
}