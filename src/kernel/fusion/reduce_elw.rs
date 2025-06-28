use crate::kernel_decl::{KernelProcedure, Kernels};

#[derive(Clone, Debug)]
pub struct ReduceExprKernelInfo {
    pub start_loc: usize,
    pub end_loc: usize,

    pub vec_size: Option<usize>,
    pub reduce_size: Option<usize>
}

pub fn fuse_rd_expr (kernel_proc: &mut KernelProcedure) {
    kernel_proc.apply(&mut |proc| {
        // ============== Track potential kernel fusion operations ============== 
        let mut elw_ks: Vec<ReduceExprKernelInfo> = vec![];
        let mut in_kernel: Option<ReduceExprKernelInfo> = None; 
        
        for (i, cmd) in proc.iter().enumerate() {
            if let Kernels::Reduce { vec_size, reduce_size, .. } = cmd {
                let new_kernel_info = ReduceExprKernelInfo { 
                    start_loc: i,
                    end_loc: i,
                    vec_size: Some(*vec_size),
                    reduce_size: Some(*reduce_size)
                };

                if let Some(kernel_info) = &mut in_kernel {
                    if kernel_info.vec_size.is_none() {
                        // Alloc/dealloc previously --> update information
                        kernel_info.vec_size = Some(*vec_size);
                        kernel_info.reduce_size = Some(*reduce_size);
                        kernel_info.end_loc += 1;
                    } else {
                        // Reduce followed by another reduce (update to latest reduce)
                        in_kernel = Some(new_kernel_info)
                    }
                }
                else {
                    // Start kernel as we start with reduce
                    in_kernel = Some(new_kernel_info)
                }
            }
            else if let Kernels::ElwExpr { size, .. } = cmd {
                // in current kernel with reduce previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches reduce dimensions
                    if kernel_info.vec_size.is_some() && kernel_info.vec_size.unwrap() == *size {
                        // push to elw ks 
                        kernel_info.end_loc += 1;
                        elw_ks.push(kernel_info.clone())
                    }

                    // reset in kernel
                    in_kernel = None;
                }
            }
            else if let Kernels::Binary { size, .. } = cmd {
                // in current kernel with reduce previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches reduce dimensions
                    if kernel_info.vec_size.is_some() && kernel_info.vec_size.unwrap() == *size {
                        // push to elw ks 
                        kernel_info.end_loc += 1;
                        elw_ks.push(kernel_info.clone())
                    }

                    // reset in kernel
                    in_kernel = None;
                }
            }
            else if let Kernels::Unary { size, .. } = cmd {
                // in current kernel with reduce previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches reduce dimensions
                    if kernel_info.vec_size.is_some() && kernel_info.vec_size.unwrap() == *size {
                        // push to elw ks 
                        kernel_info.end_loc += 1;
                        elw_ks.push(kernel_info.clone())
                    }

                    // reset in kernel
                    in_kernel = None;
                }
            }
            else if let Kernels::Dealloc { .. } = cmd {
                if let Some(current_kernel) = in_kernel.as_mut() {
                    current_kernel.end_loc += 1;
                } else {
                    in_kernel = Some(ReduceExprKernelInfo { 
                        start_loc: i,
                        end_loc: i,
                        vec_size: None,
                        reduce_size: None
                    })
                }
            }
            else if let Kernels::Alloc { .. } = cmd {
                if let Some(current_kernel) = in_kernel.as_mut() {
                    current_kernel.end_loc += 1;
                } else {
                    in_kernel = Some(ReduceExprKernelInfo { 
                        start_loc: i,
                        end_loc: i,
                        vec_size: None,
                        reduce_size: None
                    })
                }
            }
            else {
                in_kernel = None;
            }
        }

        // ============ Delete operations and replace ============ 
        let mut to_delete = 0;
        for fused_info in elw_ks {
            let mut to_insert: Vec<Kernels> = vec![];
            let size = fused_info.end_loc - fused_info.start_loc;
            for _ in 0..=size {
                let cmd = proc.remove(fused_info.start_loc - to_delete);

                if let Kernels::ElwExpr { kernels, .. } = cmd {
                    to_insert.extend(kernels);
                } else {
                    to_insert.push(cmd);
                }
            }

            proc.insert(
                fused_info.start_loc - to_delete,
                Kernels::ReduceElwExpr { 
                    kernels: to_insert,
                    reduce_size: fused_info.reduce_size.unwrap(),
                    vec_size: fused_info.vec_size.unwrap()
                }
            );

            to_delete += size;
        }

    });
}