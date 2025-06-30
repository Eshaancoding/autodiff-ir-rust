// In general, any expression that is applied elementwise
// This could be unary, binary, etc. etc.

use crate::kernel_decl::{KernelProcedure, Kernels};

#[derive(Clone, Debug)]
pub struct ElwExprKernelInfo {
    pub size: Option<usize>,
    pub start_loc: usize,
    pub end_loc: usize,
    pub num_cmds: usize,
}

// fuse operations heavily rely on prox_rev_opt and prox_opt and grouping operations together!
pub fn fuse_elw_expr (kernel_proc: &mut KernelProcedure, kernel_id: &mut usize) {
    kernel_proc.apply(&mut |proc| {
        // ============== Track potential kernel fusion operations ============== 
        let mut elw_ks: Vec<ElwExprKernelInfo> = vec![];        
        let mut in_kernel: Option<ElwExprKernelInfo> = None;

        for (i, cmd) in proc.iter().enumerate() {
            let mut append = false;
            let mut create: Option<usize> = None;

            let mut add_to_kernel = |size: &usize| {
                if let Some(current_kernel) = in_kernel.as_mut() {
                        // in a potential elw kernel 
                    if let Some(kernel_size) = current_kernel.size {
                        // same kernel size --> append
                        if kernel_size == *size {
                            current_kernel.end_loc += 1;
                            current_kernel.num_cmds += 1;
                        }
                        // different kernel size than the current pot elw kernel --> try to append to elw_ks and restart
                        else {
                            append = true;

                            // create a potentially new binary kernel (covering case of two different kernel groupings that are right next to each other)
                            create = Some(*size); 
                        }
                    } 
                    // in a potential elw kernel, but it's not yet defined as it's started by an alloc/dealloc
                    else {
                        current_kernel.end_loc += 1;
                        current_kernel.num_cmds += 1;
                        current_kernel.size = Some(*size);
                    }
                } else {
                    // not in pot kernel, create
                    in_kernel = Some(ElwExprKernelInfo { 
                        size: Some(*size), 
                        start_loc: i,
                        end_loc: i,
                        num_cmds: 1
                    });
                }
            };

            if let Kernels::Binary { size, .. } = cmd {
                add_to_kernel(size);
            }
            else if let Kernels::Unary { size, .. } = cmd {
                add_to_kernel(size);
            }
            else if let Kernels::Movement { size, .. } = cmd {
                add_to_kernel(size);
            }
            // try to encapsulate any allocs or deallocs as well
            // it's possible that we can apply any allocs/deallocs optimizations inside these fused kernels
            // if it's not possible, then the allocs and deallocs are moved outside (done at a later optimization)
            else if let Kernels::Alloc { .. } = cmd {
                if let Some(current_kernel) = in_kernel.as_mut() {
                    current_kernel.end_loc += 1; 
                }
                else {
                    in_kernel = Some(ElwExprKernelInfo { 
                        size: None, 
                        start_loc: i,
                        end_loc: i,
                        num_cmds: 0,
                    });
                }
            }
            else if let Kernels::Dealloc { .. } = cmd {
                if let Some(current_kernel) = in_kernel.as_mut() {
                    current_kernel.end_loc += 1; 
                }
                else {
                    in_kernel = Some(ElwExprKernelInfo { 
                        size: None, 
                        start_loc: i,
                        end_loc: i,
                        num_cmds: 0,
                    });
                }
            } 
            else {
                append = true;
            }

            if append {
                if let Some(current_kernel) = in_kernel.as_ref() {
                    if current_kernel.num_cmds > 1 {
                        elw_ks.push(current_kernel.clone());
                    }
                }
                in_kernel = if let Some(s) = create {
                    Some(ElwExprKernelInfo { 
                        size: Some(s), 
                        start_loc: i, 
                        end_loc: i, 
                        num_cmds: 1,
                    })
                } else {
                    None
                }
            }
        }

        if let Some(current_kernel) = in_kernel.as_ref() {
            if current_kernel.num_cmds > 1 && current_kernel.size.is_some() {
                elw_ks.push(current_kernel.clone());
            }
        }

        // ============== Delete operations and replace ============== 
        let mut to_delete = 0;
        for fused_info in elw_ks {
            let mut to_insert: Vec<Kernels> = vec![];
            let size = fused_info.end_loc-fused_info.start_loc;
            for _ in 0..=size {
                to_insert.push(proc.remove(fused_info.start_loc - to_delete));
            }

            proc.insert(
                fused_info.start_loc - to_delete, 
                Kernels::ElwExpr { kernels: to_insert, size: fused_info.size.unwrap(), id: *kernel_id }
            );

            *kernel_id += 1;
            to_delete += size;
        }
    });
}