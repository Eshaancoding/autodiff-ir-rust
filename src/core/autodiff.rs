use std::collections::HashSet;

use rand::rng;
use rand_distr::{Normal, Distribution};

pub use crate::graph::data::concat::concat;
pub use crate::graph::linear_alg::dot_product::dot;
pub use crate::devices;
use crate::{core::add_to_dep, ir::optimizations::optimize::*};

pub use crate::{
    Device, 
    Tensor, 
    TensorNode, 
    Value, 
    DEVICE, 
    nn,
    SeqF
};

pub use super::control::*;

use super::{ir_b_add, ir_b_execute, ir_b_main_block, is_harsh, set_harsh_dep_list, ConstantNode, IRBase, DEP_TRACKER, IRB};
use crate::IRCmds::{Heading, Subheading, EX};

// new tensor
pub fn tensor (data: Vec<f64>, dim: Vec<usize>) -> Tensor {
    if data.len() != dim.iter().product::<usize>() {
        panic!("# of values doesn't match dim size when calling from_vec");
    }
    let v = Value::new(data, dim);

    // add to dependency tracker
    if !is_harsh() {
        add_to_dep(v.id.clone());
    }

    Tensor::new(TensorNode {
        v,
        gd: None
    })
} 

pub fn scalar (val: f64) -> Tensor {
    tensor(vec![val], vec![1])
}

pub fn constant (data: Vec<f64>, dim: Vec<usize>) -> Tensor {
    if data.len() != dim.iter().product::<usize>() {
        panic!("# of values doesn't match dim size when calling from_vec");
    }

    let v = Value::new(data, dim);
    Tensor::new(ConstantNode {
        v 
    })
}

pub fn const_val (val: f64) -> Tensor {
    constant(vec![val], vec![1]) 
}

// helper func
pub fn ones (dim: Vec<usize>) -> Tensor {
    fill(1.0, dim) 
}

pub fn zeros (dim: Vec<usize>) -> Tensor {
    fill(0.0, dim) 
}

pub fn empty_tensor () -> Tensor {
    Tensor::new(TensorNode {
        v: Value::empty(),
        gd: None
    })
}

pub fn fill (val: f64, dim: Vec<usize>) -> Tensor {
    tensor(
        vec![val; dim.iter().product()],
        dim.clone()
    )    
}

pub fn randn (dim: Vec<usize>) -> Tensor {
    let mut rng = rng();
    let normal = Normal::new(0.0, 1.0).unwrap(); // Mean = 0, Std = 1
    let samples: Vec<f64> = (0..dim.iter().product()).map(|_| normal.sample(&mut rng)).collect();   
    tensor(samples, dim)
}

pub fn set_device <T: Device + Send + Sync + 'static> (device: T) {
    // set base
    let mut guard = IRB.lock().expect("Can't lock IR builder");
    *guard = Some(IRBase::new());
    drop(guard);

    // set custom ir builder
    let mut guard = DEVICE.lock().expect("Can't lock IR builder");
    *guard = Some(Box::new(device));
    drop(guard);

    // set dep tracker
    let mut guard = DEP_TRACKER.lock().expect("Can't lock DEP tracker");
    *guard = Some(HashSet::new());
    drop(guard);
}

/**
 * Note that ir_print will show EXIT after execute()
 */
pub fn ir_print () {
    let mut guard = IRB.lock().unwrap();
    let v = guard.as_mut().expect("Can't unpack IRBuilder");
    v.print();
}

pub fn add_heading (cmt: &str) {
    ir_b_add(Heading {
        cmt: cmt.to_string()
    });
}

pub fn add_subheading (h: &str, cmt: &str) {
    ir_b_add(Subheading { 
        h: Some(h.to_string()),
        cmt: cmt.to_string()
    });
}

pub fn add_all_subheading (cmt: &str) {
    ir_b_add(Subheading { 
        h: None,
        cmt: cmt.to_string()
    });
}

pub fn execute () {
    ir_b_main_block();
    ir_b_add(EX); // add exit
    ir_optimize();
    
    ir_b_execute();    // execute
}


/*
 Sets eager dependency optimization
 * We assume that values and gradients declared by users (`autodiff::tensor`) and will be eventually read by the user
 * Therefore, the dependency optimizer won't remove these variables. Setting eager opt will not assume this condition. 
 * Setting eager dependency opt means that every tensor being read will need to have a `keep()` before executing
 */
pub fn eager_dep_opt () {
    set_harsh_dep_list();
}