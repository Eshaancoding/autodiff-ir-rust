use crate::kernel_decl::{KernelProcedure, Kernels};

pub fn alloc_in (procedure: &mut KernelProcedure) {
    procedure.apply(&mut |f| {
        for i in 0..(f.len()-1) {
            let mut idx_to_del = None;

            if i == f.len() { break; }

            if let Kernels::Alloc { .. } = f.get(i).unwrap() {
                if f.get(i+1).unwrap().fus_get_kernels().is_some() {
                    idx_to_del = Some(i);
                }
            }

            if let Some(i) = idx_to_del {
                let k = f.remove(i);
                if let Some(block) = f.get_mut(i).unwrap().fus_get_mut_kernels() {
                    block.insert(0, k);
                }
            }
        }
    });
}