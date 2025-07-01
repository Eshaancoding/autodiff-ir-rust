use std::collections::HashMap;
use tensor_rs::{tensor_impl::gen_tensor::GenTensor, tensor_trait::{elemwise::ElemwiseTensorOp, index_slicing::IndexSlicing, reduction::ReduceTensor}};
use crate::IRCmds;

// Determine which tensor should be swapped for right broadcast (because tensor_rs is only supports right-only broadcast)
fn is_sw (a: &Vec<usize>, b: &Vec<usize>) -> bool {
    if a.len() == b.len() {
        a.iter().sum::<usize>() < b.iter().sum::<usize>()
    } else {
        a.len() < b.len()
    }
}

pub fn exec (cmd: &IRCmds, hms: &mut HashMap<String, GenTensor<f32>>) {
    match cmd {
        IRCmds::CreateMat { contents, dim, id } => {
            hms.insert(
                id.clone(),
                GenTensor::<f32>::new_raw(contents, dim)
            );
        },
        IRCmds::CreateConstant { contents, id, .. } => {
            hms.insert(
                id.clone(),
                GenTensor::<f32>::new_raw(&vec![*contents], &vec![1])
            );
        }
        IRCmds::ElwMultiply { a, b, res } => {
            let a = hms.get(a).unwrap();
            let b = hms.get(b).unwrap();
            let is_swap = is_sw(&a.size(), &b.size());
            
            hms.insert(
                res.clone(),
                if is_swap { b.mul(a) } else { a.mul(b) }
            );
        },
        IRCmds::ElwAdd { a, b, res } => {
            let a = hms.get(a).unwrap();
            let b = hms.get(b).unwrap();
            let is_swap = is_sw(&a.size(), &b.size());
            
            hms.insert(
                res.clone(),
                if is_swap { b.add(a) } else { a.add(b) }
            );
        },
        IRCmds::ElwMultiplyEq { s, o } => {
            let a = hms.get(s).unwrap();
            let b = hms.get(o).unwrap();
            let is_swap = is_sw(&a.size(), &b.size());

            hms.insert(
                s.clone(),
                if is_swap { b.mul(a) } else { a.mul(b) }
            );
        },
        IRCmds::ElwAddEq { s, o } => {
            let a = hms.get(s).unwrap();
            let b = hms.get(o).unwrap();
            let is_swap = is_sw(&a.size(), &b.size());

            *hms.get_mut(s).unwrap() = if is_swap { b.add(a) } else { a.add(b) };
        },
        IRCmds::EqualZero { a, res } => {
            let a = hms.get(a).unwrap();
            let zeros = GenTensor::<f32>::zeros_like(a);

            hms.insert(
                res.clone(),
                a.eq_t(&zeros)
            );
        },
        IRCmds::MoreZero { a, res } => {
            let a = hms.get(a).unwrap();
            let zeros = GenTensor::<f32>::zeros_like(a);

            hms.insert(
                res.clone(),
                a.gt(&zeros)
            );
        },
        IRCmds::LessZero { a, res } => {
            let a = hms.get(a).unwrap();
            let zeros = GenTensor::<f32>::zeros_like(a);

            hms.insert(
                res.clone(),
                a.lt(&zeros)
            );
        },

        IRCmds::DotProduct { a, b, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().mm(hms.get(b).unwrap())
            );
        },
        
        IRCmds::View { a, target_dim, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().reshape(target_dim)
            );
        },
        IRCmds::Index { a, index, dim, res } => {
            let dim_idx = GenTensor::<f32>::new_raw(&[*index as f32], &[1]);
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().index_select(
                    *dim,
                    &dim_idx
                ).squeeze(Some(*index))
            );
        },
        IRCmds::Recip { a, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().reciprocal()
            );
        },
        IRCmds::Sqrt { a, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().sqrt()
            );
        },
        IRCmds::Sum { a, res } => {
            let a = hms.get(a).unwrap();
            let a_dim = a.size();
            hms.insert(
                res.clone(),
                a.sum(Some(&[a_dim.len()-1]), false)
            );
        },
        IRCmds::Concat { a, b, dim, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().cat(
                    &[hms.get(b).unwrap().clone()],
                    *dim
                )
            );
        },
        IRCmds::Permute { a, p, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().permute(p)
            );
        },
        IRCmds::Exp2 { a, res } => {
            // calculate e^(ln(2)*x) since there's no exp2
            let b_times = GenTensor::<f32>::new_raw(
                &[2.0f32.ln()],
                &[1]
            );

            hms.insert(
                res.clone(),
                hms.get(a).unwrap().mul(&b_times).exp()
            );
        },
        IRCmds::Sin { a, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().sin()
            );
        },
        IRCmds::Log2 { a, res } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().log2()
            );
        },
        IRCmds::Broadcast { a, res, dim, r } => {
            // have to repeat insteda of broadcasting
            // tensor rs has right hand broadcasting, but it's too finicky to incorperate into this.
            // This is a naive CPU implementation... use OpenCL or other backends

            let a = hms.get(a).unwrap();
            let a_dim_len = a.size().len();
            let mut dim_repeat: Vec<usize> = vec![];
            for _ in 0..a_dim_len { dim_repeat.push(1); }
            dim_repeat[*dim] = *r;

            hms.insert(
                res.clone(),
                a.repeat(&dim_repeat) 
            );
        },
        IRCmds::Contigious { .. } => {
            // don't do anything. Every operation is anyways expanded to be contigious anyways
        },
        IRCmds::While { .. } => {}, // while are handled by execute
        IRCmds::If { .. } => {},    // If statements are handled by execute
        IRCmds::EX { } => {},       // exit statements are handled by execute
        IRCmds::Heading { .. } => {},
    }     
}