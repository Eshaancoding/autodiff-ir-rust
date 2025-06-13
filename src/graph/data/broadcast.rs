use crate::{ir_b_add, ir_b_id, NodeTrait, Tensor, Value};

#[derive(Clone)]
pub struct BroadcastNode {
    parent: Tensor,
    repeat: usize,
    dim: usize,
    val: Option<Value>
}

impl NodeTrait for BroadcastNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val.clone() { 
            return v;
        } 
        let p_val = self.parent.forward();
        let res_val = c_broadcast(&p_val, self.repeat, self.dim);
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        let mut p_dim = self.parent.dim();
        p_dim[self.dim] = self.repeat;
        p_dim
    }

    fn backward (&mut self, grad:Value) {
        self.parent.n.borrow_mut().backward(
            grad.to_node().sum(self.dim as i32).forward()
        );
    }      

    fn val (&self) -> Value {
        self.val.clone().expect("Run forward pass before val")
    }
}

fn is_broadcastable (dim_a: &Vec<usize>, dim_b: &Vec<usize>) -> bool {
    let dim_a_len = dim_a.len();
    let dim_b_len = dim_b.len();
    let max_dim = if dim_b_len > dim_a_len {dim_b_len} else {dim_a_len};

    if dim_a_len == 0 || dim_b_len == 0 {
        return false;
    }

    for i in 0..max_dim {
        if i >= dim_a_len || i >= dim_b_len {
            return true; // one of them is empty; therefore is broadcastable
        }

        let dim_a_access = dim_a_len - i - 1;
        let dim_b_access = dim_b_len - i - 1; 

        if dim_a[dim_a_access] != dim_b[dim_b_access] && dim_a[dim_a_access] != 1 && dim_b[dim_b_access] != 1 {
            return false;
        }
    }
        
    true 
}

/**
// (5,2,3,2) -(broadcastnode)-> (2,3,2) -(broadcastnode)-> (3,2) --> orig tens

internally, it would look like
.get(vec![5,2,3,2])
param.remove(0) # remove first dim
[2,3,2] result
then pass that to the value

.get() converts from matrix node --> actual node

internally it looks like:
let mut index = 0;
for i in 0..self.d.len() {
    ret.d.push(closure(&self.d[i], &o.d[index]));
    index += 1;
    if index >= o.d.len() {
        index = 0;
    }
}
 */

fn c_broadcast (p: &Value, r: usize, dim: usize) -> Value {
    // check if broadcastable
    let id = ir_b_id();    
    ir_b_add(crate::IRCmds::Broadcast { 
        a: p.id.clone(), 
        r,
        dim,
        res: id.clone()
    });

    let mut r_dim = p.dim.clone();
    r_dim[dim] = r;

    Value {
        dim: r_dim,
        id
    }
}

fn make_broadcast_node (n: &Tensor, target_dim: &Vec<usize>) -> Tensor {
    let mut n_dim = n.dim().clone();
    assert!(target_dim.len() >= n_dim.len(), "Wrong broadcasted node");

    for _ in 0..(target_dim.len() - n_dim.len()) {
        n_dim.insert(0, 1);
    }

    // everything is asserted
    let mut ret_n = n.clone().view(
        {
            let dim: &Vec<usize> = &n_dim;
            dim.iter().map(|&i| i as i32).collect::<Vec<i32>>()
        }
    );

    for i in 0..target_dim.len() {
        if target_dim[i] != n_dim[i] {
            assert_eq!(n_dim[i], 1, "Can't broadcast to target dim");
            ret_n = Tensor::new(
                BroadcastNode {
                    parent: ret_n,
                    repeat: target_dim[i],
                    dim: i,
                    val: None
                }
            );
        } 
    }

    ret_n
}

// tries to broadcast both values. 
// If communicative, then it will prioritize right-broadcasting in favor for left-broadcasting
pub fn try_broadcast (a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    // if empty dim on a or b, fill to [1] 
    let a = if a.dim().len() == 0 && b.dim().len() > 0 {
        a.clone().unsqueeze(0)
    } else { 
        a.clone() 
    };

    let b = if b.dim().len() == 0 && a.dim().len() > 0 {
        b.clone().unsqueeze(0)
    } else { 
        b.clone() 
    };

    if a.dim() == b.dim() {
        return (a.clone(), b.clone())
    }     

    // If one of the values are single value constants, don't broadcast
    // During kernel conversion will automatically refill them as constants and inline them.
    if a.is_const() || b.is_const() {
        return (a.clone(), b.clone())
    }

    // if either are constant, then you can just multiply directly
    let is_b = is_broadcastable(&a.dim(), &b.dim());

    if is_b {
        // find which to broadcast to
        let a_dim_len = a.dim().len();
        let b_dim_len = b.dim().len();

        let is_a_broadcast = if a_dim_len == b_dim_len {
            a.dim().iter().sum::<usize>() < b.dim().iter().sum::<usize>()
        } else {
            a.dim().len() < b.dim().len()
        };

        if is_a_broadcast {
            return (make_broadcast_node(&a, &b.dim()), b.clone())
        } else {
            return (a.clone(), make_broadcast_node(&b, &a.dim()))
        } 
    } 
    panic!("Cannot broadcast {:?} to {:?}", a.dim(), b.dim());
}

impl Tensor {
    // nodes can be explicitly broadcast if needed
    pub fn broadcast (&self, dim: i32, r: usize) -> Tensor {
        let mut target_dim = self.dim().clone();
        let dim = if dim < 0 {
            ((target_dim.len() as i32) + dim) as usize
        } else {
            dim as usize     
        };
                
        target_dim[dim] = r;

        make_broadcast_node(&self, &target_dim) 
    }
}