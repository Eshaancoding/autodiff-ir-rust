use crate::{ir_b_add, ir_b_id, Tensor, NodeTrait, Value, IRCmds};

#[derive(Clone)]
pub struct SumNode {
    parent: Tensor,
    val: Option<Value>
}

impl NodeTrait for SumNode {
    fn forward (&mut self) -> Value {
        if let Some(v) = self.val() { 
            return v;
        }       
        
        let val = self.parent.forward();
        assert_eq!(self.parent.dim().len(), 2, "Reduce (sum) node needs dim=2");

        let res_val = c_sum(&val, self.val.as_ref().map(|v| v.id.clone()));
        self.val = Some(res_val.clone());
        res_val
    }

    fn dim (&self) -> Vec<usize> {
        let mut d = self.parent.dim().clone();
        d.remove(d.len()-1);
        d
    }

    fn backward (&mut self, grad: Value) {
        let repeat_n = *self.parent.dim().last().unwrap();
        
        self.parent.n.borrow_mut().backward(
            grad.to_node().
                unsqueeze(-1).
                broadcast(-1, repeat_n)
            .forward()
        ); 
    }

    fn val (&self) -> Option<Value> {
        if self.parent.val().is_none() { return None }
        self.val.clone()
    }

    fn deep_copy (&self) -> Box<dyn NodeTrait> {
        Box::new(self.clone())
    }
}

impl Tensor {
    pub fn sum (&self, dim:i32) -> Tensor {
        let p_dim = self.dim().len();        

        let dim = if dim < 0 { (p_dim as i32 + dim) as usize } else { dim as usize };
        
        // ========== permute to last dim ========== 
        // 0 1 2 3 4 --> 0 1 4 3 2
        // dim=2

        // 0 1 4 3 2 --> 0 1 3 2
        let mut permute_to: Vec<usize> = (0..p_dim).collect();
        *permute_to.last_mut().unwrap() = dim;
        permute_to[dim] = p_dim-1;

        let node = self.permute(&permute_to);

        // ========== View last dim ========== 
        let mut new_dim = node.dim();
        let node = node.view(vec![-1, *new_dim.last().unwrap() as i32]);
        
        // ========== Sum node ========== 
        let node = Tensor::new(SumNode {
            parent: node,
            val: None
        });

        // ========== Unview ========== 
        new_dim.remove(new_dim.len()-1);
        let node = node.view(new_dim.iter().map(|&v| v as i32).collect());

        // ========== Unpermute ========== 
        permute_to.remove(dim);
        let node = node.permute(&permute_to);

        node
    }

    pub fn mean (&self, dim:i32) -> Tensor {
        let p_dim = self.dim();
        let dim = if dim < 0 { p_dim.len() as i32 + dim } else { dim };
        let orig_dim = p_dim.get(dim as usize).unwrap().clone();
        self.sum(dim) / (orig_dim as f32)
    }

    pub fn var (&self, dim:i32, correction:usize) -> Tensor {
        let a = (self.clone() - self.mean(dim).unsqueeze(dim)).pow2();

        let p_dim = self.dim();
        let dim_size = if dim < 0 { p_dim.len() as i32 + dim } else { dim } as usize;
        let div = p_dim.get(dim_size).unwrap();

        a.sum(dim) / (div - correction) as f32
    }
}

fn c_sum (a: &Value, id: Option<String>) -> Value {
    let id = id.or_else(|| Some(ir_b_id()) ).unwrap();
    ir_b_add(IRCmds::Sum { 
        a: a.id.clone(), 
        res: id.clone()
    });

    let mut d = a.dim.clone();
    d.remove(d.len()-1);

    Value { 
        dim: d, 
        id
    }
}