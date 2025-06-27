use crate::kernel_decl::{KernelProcedure, Kernels};

#[derive(Clone, Debug)]
pub struct DpExprKernelInfo {
    pub start_loc: usize,
    pub end_loc: usize,

    pub batch_size: Option<usize>,
    pub input_size: Option<usize>,
    pub output_size: Option<usize>,
}

pub fn fuse_dp_expr (kernel_proc: &mut KernelProcedure) {
    kernel_proc.apply(&mut |proc| {
        // ============== Track potential kernel fusion operations ============== 
        let mut elw_ks: Vec<DpExprKernelInfo> = vec![];
        let mut in_kernel: Option<DpExprKernelInfo> = None; 
        
        for (i, cmd) in proc.iter().enumerate() {
            if let Kernels::DotProd { batch_size, input_size, output_size, .. } = cmd {
                let new_kernel_info = DpExprKernelInfo { 
                    start_loc: i,
                    end_loc: i,
                    batch_size: Some(*batch_size),
                    input_size: Some(*input_size),
                    output_size: Some(*output_size)
                };

                if let Some(kernel_info) = &mut in_kernel {
                    if kernel_info.batch_size.is_none() {
                        // Alloc/dealloc previously --> update information
                        kernel_info.batch_size = Some(*batch_size);
                        kernel_info.input_size = Some(*input_size);
                        kernel_info.output_size = Some(*output_size);
                        kernel_info.end_loc += 1;
                    } else {
                        // Dot product followed by another dot product (update to latest dp)
                        in_kernel = Some(new_kernel_info)
                    }
                }
                else {
                    // Start kernel as we start with dot product
                    in_kernel = Some(new_kernel_info)
                }
            }
            else if let Kernels::ElwExpr { size, .. } = cmd {
                // in current kernel with dot product previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches dot product dimensions
                    if kernel_info.batch_size.is_some() && kernel_info.batch_size.unwrap() * kernel_info.output_size.unwrap() == *size {
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
                    in_kernel = Some(DpExprKernelInfo { 
                        start_loc: i,
                        end_loc: i,
                        batch_size: None,
                        input_size: None,
                        output_size: None 
                    })
                }
            }
            else if let Kernels::Alloc { .. } = cmd {
                if let Some(current_kernel) = in_kernel.as_mut() {
                    current_kernel.end_loc += 1;
                } else {
                    in_kernel = Some(DpExprKernelInfo { 
                        start_loc: i,
                        end_loc: i,
                        batch_size: None,
                        input_size: None,
                        output_size: None 
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
                Kernels::DPElwExpr { 
                    kernels: to_insert,
                    batch_size: fused_info.batch_size.unwrap(),
                    input_size: fused_info.input_size.unwrap(),
                    output_size: fused_info.output_size.unwrap()
                }
            );

            to_delete += size;
        }

    });
}