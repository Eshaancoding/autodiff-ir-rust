// metainfo that is helpful to device
use crate::kernel_decl::{Input, Kernels, Output};

pub fn get_inputs_fused (kernels: &Vec<Kernels>) -> Vec<&Input> {
    let v: Vec<&Input> = kernels.iter()
        .map(|v| v.get_inputs() )
        .collect::<Vec<_>>()
        .concat();

    v
}

pub fn get_outputs_fused (kernels: &Vec<Kernels>) -> Vec<&Output> {
    let v: Vec<&Output> = kernels.iter()
        .map(|v| v.get_outputs() )
        .collect::<Vec<_>>()
        .concat();

    v
}

impl Kernels {
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

    pub fn get_outputs (&self) -> Vec<&Output> {
        match self {
            Kernels::Unary { res, .. } => vec![res],
            Kernels::Reduce { res, .. } => vec![res],
            Kernels::Movement { res, .. } => vec![res],
            Kernels::Binary { res, .. } => vec![res],
            Kernels::DotProd { res, .. } => vec![res],

            Kernels::DPElwExpr { kernels, .. } => get_outputs_fused(kernels),
            Kernels::ElwExpr { kernels, .. } => get_outputs_fused(kernels),
            Kernels::ReduceElwExpr { kernels, .. } => get_outputs_fused(kernels),

            _ => vec![]
        } 
    }
} 