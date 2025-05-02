use crate::core::{autodiff, node::Tensor};


impl Tensor {
    // apply function across every node --> concat
    // technically, there is a permute at the end, so
    // every node --> apply func --> permute to last dim --> concat last dim --> permute to orig dim
    pub fn repeat<F> (&self, f: F, len: usize, dim: i32) -> Tensor 
        where F: Fn(&Tensor, usize) -> Tensor
    {
        let mut vector_nodes: Vec<Tensor> = vec![];
        for i in 0..len {
            vector_nodes.push(f(self, i).unsqueeze(dim)); 
        }
        
        autodiff::concat(vector_nodes, dim)
    }
}