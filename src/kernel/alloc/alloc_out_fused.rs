use crate::kernel_decl::{KernelProcedure, Kernels};

/*
(3): DP + Elw Kernel Fusion (2x64 DP 64x128) -(elw)-> 256

    Alloc ad 256
    M (id: ad, access: ((#x << 7) + #y))  =  M (id: ab, access: ((#x << 6) + #y))  (2x64 DP 64x128) M (id: c, access: ((#y << 6) + #x))
    M (id: ad, access: #global)  =  CS (V: 1)  Multiply (256)  M (id: ad, access: #global)
    M (id: ad, access: #global)  =  M (id: ad, access: #global)  Multiply (256)  M (id: v, access: #global)
    Dealloc v 256
    M (id: ad, access: #global)  =  M (id: ad, access: #global)  Multiply (256)  CS (V: 0.6931471805599453)
    M (id: ad, access: #global)  =  M (id: ad, access: #global)  Multiply (256)  M (id: o, access: #global)
    Dealloc o 256
    M (id: ad, access: #global)  =  CS (V: 1.4426950408889634)  Multiply (256)  M (id: ad, access: #global)
    M (id: ad, access: #global)  =  CS (V: -1)  Multiply (256)  M (id: ad, access: #global)

This function takes the allocations out of the fusion operations
*/

pub fn alloc_out_fused (kernel_proc: &mut KernelProcedure) {
    kernel_proc.step_cmd(&mut |proc, i| {
        let curr_cmd = proc.get_mut(*i).unwrap();

        let mut begin: Vec<Kernels> = vec![];
        let mut end: Vec<Kernels> = vec![];
        if let Some(fused_kernels) = curr_cmd.fus_get_mut_kernels() {
            // get the first few alloc/dealloc
            let mut begin_size = 0; 
            for cmd in fused_kernels.iter() {
                match cmd {
                    Kernels::Alloc { .. } => { begin_size += 1 },
                    Kernels::Dealloc { .. } => { begin_size += 1 },
                    _ => { break; }
                }
            }
            
            // remove first few ending alloc
            for _ in 0..begin_size {
                begin.push(fused_kernels.remove(0));
            }

            // get the last few alloc/dealloc
            let mut end_size = 0; 
            for cmd in fused_kernels.iter().rev() {
                match cmd { 
                    Kernels::Alloc { .. } => { end_size += 1 },
                    Kernels::Dealloc { .. } => { end_size += 1 },
                    _ => { break; }
                }
            }

            // remove the last few ending 
            for _ in 0..end_size {
                end.push(fused_kernels.remove(fused_kernels.len() - end_size))
            }

            // find locations 
            let mut locs: Vec<(usize, bool)> = vec![];
            for (idx, cmd) in fused_kernels.iter().enumerate() {
                match cmd {
                    Kernels::Alloc { .. } => { locs.push((idx, true)); },
                    Kernels::Dealloc { .. } => { locs.push((idx, false)); },
                    _ => { }
                }
            }

            // delete --> insert to begin and end
            let mut delete = 0;
            for (loc, is_alloc) in locs {
                let k = fused_kernels.remove(loc - delete);

                if is_alloc {
                    begin.push(k);
                } else {
                    end.push(k)
                } 
                delete += 1;
            }
        } 

        // actually insert at the main proc
        for e in end {
            proc.insert(*i + 1, e);
        }
        for b in begin {
            proc.insert(*i, b);
        } 
        
        true
    }) 


}