// metainfo that is helpful to device
use crate::kernel_decl::{Input, Kernels};

pub fn get_inputs_fused (kernels: &Vec<Kernels>) -> Vec<&Input> {
    let v: Vec<&Input> = kernels.iter()
        .map(|v| v.get_inputs() )
        .collect::<Vec<_>>()
        .concat();

    v
}

impl Kernels {
    pub fn get_kernel_work_size (&self) -> Option<usize> {
        match self {
            Kernels::Unary { size, .. } => { Some(*size) },
            Kernels::Binary { size, .. } => { Some(*size) },
            Kernels::Reduce { vec_size, reduce_size, .. } => { Some(*vec_size * *reduce_size) },
            Kernels::DotProd { res_shape, ..} => { Some(res_shape.0 * res_shape.1) },
            Kernels::Movement { size, .. } => { Some(*size) },
            Kernels::ElwExpr { size, .. } => { Some(*size) },
            Kernels::DPElwExpr { res_shape, .. } => { Some(res_shape.0 * res_shape.1) },
            Kernels::ReduceElwExpr { vec_size, reduce_size, .. } => { Some(*vec_size * *reduce_size) },
            _ => None,
        }
    }

    pub fn get_kernel_id (&self) -> Option<usize> {
        match self {
            Kernels::Unary { id, .. } => Some(*id),
            Kernels::Binary { id, .. } => Some(*id),
            Kernels::Reduce { id, .. } => Some(*id),
            Kernels::DotProd { id, .. } => Some(*id),
            Kernels::Movement { id, .. } => Some(*id),
            Kernels::ElwExpr { id, .. } => Some(*id),
            Kernels::DPElwExpr { id, .. } => Some(*id),
            Kernels::ReduceElwExpr { id, .. } => Some(*id),
            _ => None
        }
    }

    pub fn get_inputs (&self) -> Vec<&Input> {
        match self {
            Kernels::Unary { a, .. } => vec![a],
            Kernels::Reduce { a, .. } => vec![a],
            Kernels::Movement { a, .. } => vec![a],
            Kernels::Binary { a, b, .. } => vec![a, b],
            Kernels::DotProd { a, b, .. } => vec![a, b],
            
            // note that there might be duplicated inputs
            // get_arg_list should be able to handle that 
            Kernels::DPElwExpr { kernels, .. } => get_inputs_fused(kernels),
            Kernels::ElwExpr { kernels, .. } => get_inputs_fused(kernels),
            Kernels::ReduceElwExpr { kernels, .. } => get_inputs_fused(kernels),
            _ => vec![]
        }
    }
} 