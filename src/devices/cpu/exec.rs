use indexmap::IndexMap;
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

pub fn exec (cmd: &IRCmds, hms: &mut IndexMap<String, GenTensor<f64>>) {
    match cmd {
        IRCmds::CreateMat { contents, dim, id } => {
            hms.insert(
                id.clone(),
                GenTensor::<f64>::new_raw(contents, dim)
            );
        },
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
            let zeros = GenTensor::<f64>::zeros_like(a);

            hms.insert(
                res.clone(),
                a.eq_t(&zeros)
            );
        },
        IRCmds::MoreZero { a, res } => {
            let a = hms.get(a).unwrap();
            let zeros = GenTensor::<f64>::zeros_like(a);

            hms.insert(
                res.clone(),
                a.gt(&zeros)
            );
        },
        IRCmds::LessZero { a, res } => {
            let a = hms.get(a).unwrap();
            let zeros = GenTensor::<f64>::zeros_like(a);

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
            let dim_idx = GenTensor::<f64>::new_raw(&[*index as f64], &[1]);
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().index_select(
                    *dim,
                    &dim_idx
                ).squeeze(Some(*index))
            );
        },
        IRCmds::Neg { a, res } => {
            let b_times = GenTensor::<f64>::new_raw(
                &[-1.0],
                &[1]
            );

            hms.insert(
                res.clone(),
                hms.get(a).unwrap().mul(&b_times)
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
        IRCmds::Sum { a, dim, res } => {
            if hms.get(a).is_none() {
                println!("SUM A: {}", a);
            }
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().sum(Some(&[*dim]), false)
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
            let b_times = GenTensor::<f64>::new_raw(
                &[2.0f64.ln()],
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
        IRCmds::Broadcast { a, res, dim, .. } => {
            hms.insert(
                res.clone(),
                hms.get(a).unwrap().squeeze(Some(*dim)) // tensor_rs already has right broadcasting (no left however, this may be an error in the future cases)
            );
        },
        IRCmds::EX => {},        // handled by execute
        IRCmds::BR { .. } => {}, // handled by execute
        IRCmds::BRE { .. } => {},// handled by execute
        IRCmds::BRZ { .. } => {},// handled by execute
        IRCmds::Heading { .. } => {},
        IRCmds::Subheading { .. } => {}
    }     
}