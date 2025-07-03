/*
(7): Reduce + Elw Kernel Fusion (Vec/X: 64, Reduce/Y: 2) -(elw)-> 64

    Alloc bd 64
    M (id: bd, access: #x)  =   Sum (Vec/X: 64, Reduce/Y: 2)  (M (id: ab, access: ((#y << 6) + #x)))
    Dealloc ab 128
    M (id: bd, access: #global)  =  CS (V: 0.1)  Multiply (64)  M (id: bd, access: #global)
    M (id: bd, access: #global)  =  CS (V: -1)  Multiply (64)  M (id: bd, access: #global)
    M (id: d, access: #global)  =  M (id: d, access: #global)  Add (64)  M (id: bd, access: #global)
    Dealloc bd 64

Here, bd is allocated and deallocated within this kernel fusion. We can replace to global in these conditions:

1. alloc and dealloc of the same variable 
2. All result of the var (except for the first result) must be #global
3. All deps of the var must be #global

In which case, we don't allocate the vector bd at all. We just use a single temporary storage variable within the kernel

Note this becomes extremely helpful here:

(4): DP + Elw Kernel Fusion (256x2 DP 2x128) -(elw)-> 32768
    Alloc av 32768
    M (id: av, access: ((#x << 7) + #y))  =  M (id: e, access: ((#y << 8) + #x))  (256x2 DP 2x128) M (id: ad, access: ((#x << 7) + #y))
    M (id: av, access: #global)  =  CS (V: 0.1)  Multiply (32768)  M (id: av, access: #global)
    M (id: av, access: #global)  =  CS (V: -1)  Multiply (32768)  M (id: av, access: #global)
    M (id: a, access: #global)  =  M (id: a, access: #global)  Add (32768)  M (id: av, access: #global)
    Dealloc av 32768

We don't have to allocate a gradient variable (32768-long allocation!). As soon as we compute the gradient, apply to it's weight
*/

use std::collections::HashMap;

use crate::kernel_decl::{Expression, KernelProcedure, Kernels};

pub fn alloc_temp_opt (kernel_proc: &mut KernelProcedure) {
    let f = |v: &mut Vec<Kernels>| {
        let mut pot_temps: HashMap<String, bool> = HashMap::new();
        for cmd in v.iter() {
            match cmd {
                Kernels::Alloc { id, .. } => {
                    pot_temps.insert(id.clone(), false);
                },
                Kernels::Dealloc { id, .. }  => {
                    pot_temps.get_mut(id).map(|f| { *f = true; } );
                }
                _ => {}
            }
        }

        let pot_temps: Vec<_> = pot_temps.iter()
            .filter(|f| *f.1)
            .map(|f| f.0.clone())
            .collect();

        if pot_temps.len() == 0 { return; }        
        
        // I have absolutely no cases where there might be multiple temporary variables in one kernel 
        // So far, haven't seen an example of this. However, if it does happen, then I will know ;)
        if pot_temps.len() > 1 { println!("Multiple temporary variables? Implement this") }

        let var = pot_temps.get(0).unwrap().clone();

        // check 2. and 3.
        let mut result_expr: Vec<&Expression> = vec![];
        let mut dep_expr: Vec<&Expression> = vec![];
        for cmd in v.iter() {
            if cmd.get_res().is_some_and(|f| *f == var) {
                if let Some(r) = cmd.get_res_access_expr() { result_expr.push(r); }
            }
            dep_expr.extend(cmd.get_dep_access_expr(&var))
        }

        let all_dep_global = dep_expr.iter().all(|f| f.is_global());

        result_expr.remove(0);
        let all_res_global = result_expr.iter().all(|f| f.is_global());

        if !all_dep_global || !all_res_global {
            return; // doesn't satisfy condition 2 or 3
        }

        // Replace all references of var to temporary
        for cmd in v.iter_mut() {
            if cmd.get_res().is_some_and(|f| *f == var) { cmd.change_res_to_temp(); }
            cmd.change_dep_to_temp(&var);
        }

        // delete allocs
        *v = v.iter().filter(|&v| {
            match v {
                Kernels::Alloc { id, .. } => {
                    if *id == var {
                        return false
                    }
                },
                Kernels::Dealloc { id, .. } => {
                    if *id == var {
                        return false
                    }
                },
                _ => {}
            } 
            true
        })
        .map(|v| v.clone())
        .collect();
    };

    kernel_proc.apply(&mut |proc| {
        for cmd in proc.iter_mut() {
            if let Some(kernel) = cmd.fus_get_mut_kernels() {
                f(kernel)
            }
        }
    })
}
