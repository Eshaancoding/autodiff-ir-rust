use crate::kernel_decl::{KernelProcedure, Kernels};

#[derive(Clone, Debug)]
pub struct DpExprKernelInfo {
    pub start_loc: usize,
    pub end_loc: usize,

    pub a_shape: Option<(usize, usize)>,
    pub b_shape: Option<(usize, usize)>,
    pub out_shape: Option<(usize, usize)>
}

pub fn fuse_dp_expr (kernel_proc: &mut KernelProcedure, kernel_id: &mut usize) {
    kernel_proc.apply(&mut |proc| {
        // ============== Track potential kernel fusion operations ============== 
        let mut elw_ks: Vec<DpExprKernelInfo> = vec![];
        let mut in_kernel: Option<DpExprKernelInfo> = None; 
        
        for (i, cmd) in proc.iter().enumerate() {
            if let Kernels::DotProd { a_shape, b_shape, res_shape: out_shape, .. } = cmd {
                let new_kernel_info = DpExprKernelInfo { 
                    start_loc: i,
                    end_loc: i,
                    a_shape: Some(*a_shape),
                    b_shape: Some(*b_shape),
                    out_shape: Some(*out_shape) 
                };

                if let Some(kernel_info) = &mut in_kernel {
                    if kernel_info.a_shape.is_none() {
                        // Alloc/dealloc previously --> update information
                        kernel_info.a_shape = Some(*a_shape);
                        kernel_info.b_shape = Some(*b_shape);
                        kernel_info.out_shape = Some(*out_shape);
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
                    if kernel_info.out_shape.is_some_and(|f| f.0 * f.1 == *size) {
                        // push to elw ks 
                        kernel_info.end_loc += 1;
                        elw_ks.push(kernel_info.clone())
                    }

                    // reset in kernel
                    in_kernel = None;
                }
            }
            else if let Kernels::Binary { size, .. } = cmd {
                // in current kernel with dot product previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches dot product dimensions
                    if kernel_info.out_shape.is_some_and(|f| f.0 * f.1 == *size) {
                        // push to elw ks 
                        kernel_info.end_loc += 1;
                        elw_ks.push(kernel_info.clone())
                    }

                    // reset in kernel
                    in_kernel = None;
                }
            }
            else if let Kernels::Unary { size, .. } = cmd {
                // in current kernel with dot product previously
                if let Some(kernel_info) = &mut in_kernel {
                    // elw matches dot product dimensions
                    if kernel_info.out_shape.is_some_and(|f| f.0 * f.1 == *size) {
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
                        a_shape: None,
                        b_shape: None,
                        out_shape: None 
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
                        a_shape: None,
                        b_shape: None,
                        out_shape: None 
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
                    a_shape: fused_info.a_shape.unwrap(),
                    b_shape: fused_info.b_shape.unwrap(),
                    res_shape: fused_info.out_shape.unwrap(),
                    id: *kernel_id
                }
            );

            *kernel_id += 1;
            to_delete += size;
        }

    });
}