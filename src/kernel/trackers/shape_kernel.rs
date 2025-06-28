use std::collections::HashMap;
use crate::{kernel_decl::Kernels, Device};

#[derive(Clone, Debug)]
pub struct ShapeTrackerKernel {
    shape: HashMap<String, usize>
}

impl ShapeTrackerKernel {
    pub fn new () -> ShapeTrackerKernel {
        ShapeTrackerKernel { shape: HashMap::new() }
    }

    pub fn step (&mut self, device: &dyn Device, cmd: &Kernels) {
        match cmd {
            Kernels::Alloc { id, size, .. } => {
                self.shape.insert(id.clone(), *size);
            },
            Kernels::Unary { res, size, .. } => {
                self.shape.insert(res.id.clone(), *size);
            },
            Kernels::Binary { res, size, .. } => {
                self.shape.insert(res.id.clone(), *size);
            },
            Kernels::Reduce { res, vec_size, .. } => {
                self.shape.insert(res.id.clone(), *vec_size);
            },
            Kernels::DotProd { res, batch_size, input_size, output_size , .. }  => {
                let out_shape = device.dot_prod_shape(
                    &vec![*batch_size, *input_size], 
                    &vec![*input_size, *output_size]
                );
                self.shape.insert(res.id.clone(),  out_shape.iter().product::<usize>());
            },
            Kernels::Movement { res, size, ..} => {
                self.shape.insert(res.id.clone(), *size);
            },
            _ => {} 
        }
    }

    // the shape returned is not necessarily represented by the shape in the alloc tracker
    // it tracks the general shape of the tensor, but the tensor may have a size of 1 nonetheless
    // take, for example, a A(B,M) + B(1,M). We are broadcasting from (1,M) to (B,M)
    // the shape of B will be broadcasted to (B,M) to fit A. However, the allocation will still be (1,M)
    pub fn get_shape (&self, id: &String) -> &usize {
        self.shape.get(id).expect("Unable to get shape at shape tracker alloc")
    }
}