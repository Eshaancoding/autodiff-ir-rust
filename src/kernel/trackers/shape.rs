use std::collections::HashMap;
use crate::{Device, IRCmds};

#[derive(Clone, Debug)]
pub struct ShapeTracker {
    pub shape: HashMap<String, Vec<usize>>
}

impl ShapeTracker {
    pub fn new () -> ShapeTracker {
        ShapeTracker { shape: HashMap::new() }
    }

    pub fn step (&mut self, device: &dyn Device, cmd: &IRCmds) {
        match cmd {
            IRCmds::CreateMat { dim, id, .. } => {
                self.shape.insert(
                    id.clone(),
                    dim.clone()
                );
            },
            IRCmds::CreateConstant { id, dim, .. } => {
                self.shape.insert(
                    id.clone(),
                    dim.clone()
                );
            },
            
            IRCmds::ElwMultiply { a, res, .. } => { self.shape.insert( res.clone(), self.shape.get(a).unwrap().clone() ); },
            IRCmds::ElwAdd { a, res, .. } => { self.shape.insert( res.clone(), self.shape.get(a).unwrap().clone() ); },
            IRCmds::ElwMultiplyEq { s, o, .. } => { self.shape.insert( s.clone(), self.shape.get(o).unwrap().clone() ); },
            IRCmds::ElwAddEq { s, o, .. } => { self.shape.insert( s.clone(), self.shape.get(o).unwrap().clone() ); },

            IRCmds::EqualZero { a, res, .. } => { self.shape.insert( res.clone(), self.shape.get(a).unwrap().clone() ); },
            IRCmds::MoreZero { a, res, .. } => { self.shape.insert( res.clone(), self.shape.get(a).unwrap().clone() ); },
            IRCmds::LessZero { a, res, .. } => { self.shape.insert( res.clone(), self.shape.get(a).unwrap().clone() ); },

            IRCmds::Sum { a, res} => {
                let mut copy_shape = self.shape.get(a).unwrap().clone();
                copy_shape.remove(copy_shape.len()-1);

                self.shape.insert( 
                    res.clone(),
                    copy_shape
                );
            },
            IRCmds::DotProduct { a, b, res } => {
                let a_shape = self.shape.get(a).unwrap();
                let b_shape = self.shape.get(b).unwrap();
                let res_shape = device.dot_prod_shape(a_shape, b_shape);

                self.shape.insert(
                    res.clone(),
                    res_shape
                );
            },
            IRCmds::View { target_dim, res, .. } => {
                self.shape.insert(
                    res.clone(),
                    target_dim.clone()
                );
            },
            IRCmds::Index { a, dim, res, .. } => {
                let mut copy_shape = self.shape.get(a).unwrap().clone();
                copy_shape.remove(*dim);

                self.shape.insert( 
                    res.clone(),
                    copy_shape
                );   
            },
            IRCmds::Concat { a, b, dim, res } => {
                let mut res_clone = self.shape.get(a).unwrap().clone();
                let b_shape = self.shape.get(b).unwrap();
                res_clone[*dim] += b_shape[*dim];

                self.shape.insert(
                    res.clone(), 
                    res_clone
                );
            },
            IRCmds::Permute { a, p, res } => {
                let a_shape = self.shape.get(a).unwrap();
                let mut dim = vec![0; a_shape.len()];
                for i in 0..a_shape.len() {
                    dim[i] = a_shape[p[i]];
                }

                self.shape.insert(
                    res.clone(),
                    dim
                );
            },
            IRCmds::Broadcast { a, dim, r, res } => {
                let mut res_shape = self.shape.get(a).unwrap().clone();
                res_shape[*dim] = *r;

                self.shape.insert(
                    res.clone(),
                    res_shape
                );
            },
            IRCmds::Exp2 { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); }
            IRCmds::Log2 { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); }
            IRCmds::Sin { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); }
            IRCmds::Recip { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); }
            IRCmds::Sqrt { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); }
            IRCmds::Contigious { a, res } => { self.shape.insert(res.clone(), self.shape.get(a).unwrap().clone() ); },
            _ => {}
        }
    }

    // the shape returned is not necessarily represented by the shape in the alloc tracker
    // it tracks the general shape of the tensor, but the tensor may have a size of 1 nonetheless
    // take, for example, a A(B,M) + B(1,M). We are broadcasting from (1,M) to (B,M)
    // the shape of B will be broadcasted to (B,M) to fit A. However, the allocation will still be (1,M)
    pub fn get_shape (&self, id: &String) -> &Vec<usize> {
        self.shape.get(id).expect("Unable to get shape at shape tracker")
    }
}