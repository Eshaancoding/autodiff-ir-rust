use crate::kernel_decl::{KernelProcedure, Kernels};


/*
(3): DP + Elw Kernel Fusion (2x64 DP 64x128) -(elw)-> 256
    ... 
    Alloc av 32768

(4): DP + Elw Kernel Fusion (256x2 DP 2x128) -(elw)-> 32768

    ...
    Dealloc av 32768
    Alloc ax 128

The alloc av from the last kernel fusion is better suited to next kernel fusion. 
32768 --> matches the size of (4), not the size of (3). 

This function will attempt to find these allocations and attempt to switch those allocations to the next one (if desirable)
In a way, this a form of "prox_opt" but for allocations rather than general computations 
*/


pub fn alloc_switch (kernel_proc: &mut KernelProcedure) {
    kernel_proc.apply(&mut |proc| {
        for i in 0..(proc.len()-1) {
            let next_cmd_size = proc.get(i+1).unwrap().get_elw_size_fusion();
            let current_cmd = proc.get_mut(i).unwrap();

            // get last few allocs 
            let mut last_few_allocs: Vec<(usize, usize)> = vec![];
            if let Some(list) = current_cmd.fus_get_kernels()  {
                for (idx, k) in list.iter().rev().enumerate() {
                    match k {
                        Kernels::Alloc { size, .. } => { last_few_allocs.push((*size, list.len() - idx - 1)); }, // alloc
                        Kernels::Dealloc { .. } => {}, // do nothing
                        _ => { break; } // any other operation --> stop
                    }
                }
            } 

            if let Some(s) = next_cmd_size {
                last_few_allocs = last_few_allocs.iter()
                    .filter(|&v| {v.0 == s})
                    .map(|v| *v)
                    .collect();
            } 
            else {
                continue; // next command is not a fusion operation
            }

            // no allocs at the end --> continue on to the next
            if last_few_allocs.len() == 0 { continue; }

            // we have valid alloc. Delete and then insert to the next cmd
            last_few_allocs.sort_by(|a, b| a.1.cmp(&b.1));

            // remove
            let mut rm: Vec<Kernels> = vec![];
            let mut delete = 0;
            for (_, delete_idx) in last_few_allocs {
                let cmd = current_cmd.fus_get_mut_kernels().unwrap().remove(delete_idx - delete);
                delete += 1;
                rm.push(cmd);
            }

            // add
            for r in rm {
                proc.get_mut(i+1).unwrap().fus_get_mut_kernels().unwrap().insert(0, r);
            }
        }
    });    
}