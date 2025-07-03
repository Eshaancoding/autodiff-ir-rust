// insert allocations and deallocations accordant to alloc tracker
use std::collections::HashMap;
use crate::{core::ret_dep_list, kernel_decl::{KernelProcedure, Kernels}, trackers::{AllocTracker, Location}, Device};

pub fn step_procedure<'a> (device: &dyn Device, proc: &'a KernelProcedure, alloc_tracker: &mut AllocTracker<'a>) {
    for (idx, cmd) in proc.iter().enumerate() {

        if let Kernels::While { block, .. } = cmd {
            let mut cp = alloc_tracker.clone();
            step_procedure(device, block, &mut cp);
            alloc_tracker.merge(cp, &Location {
                proc_id: proc.id.clone(),
                loc: idx+1
            });
        }
        else if let Kernels::If { conditions, else_proc } = cmd {
            for (_, block) in conditions.iter() {
                let mut cp = alloc_tracker.clone();
                step_procedure(device, block, &mut cp);
                alloc_tracker.merge(cp, &Location {
                    proc_id: proc.id.clone(),
                    loc: idx+1
                });
            }

            if let Some(block) = else_proc {
                let mut cp = alloc_tracker.clone();
                step_procedure(device, block, &mut cp);
                alloc_tracker.merge(cp, &Location {
                    proc_id: proc.id.clone(),
                    loc: idx+1
                });
            }
        }

        alloc_tracker.step(device, cmd, Location {
            proc_id: proc.id.clone(),
            loc: idx
        }); 
    } 
}

pub fn insert_alloc<'a> (device: &dyn Device, kernel_proc: &mut KernelProcedure, var_changed: &Vec<String>) {
    let dep_vars = ret_dep_list();
    let mut alloc_tracker = AllocTracker::new(&dep_vars, var_changed);
    
    // ====================== Step through kernel procedure ====================== 
    step_procedure(device, &kernel_proc, &mut alloc_tracker);

    // ====================== Insert Allocations! ====================== 
    let mut total_list: Vec<_> = alloc_tracker.vars.iter()
        .map(|(_, v)| {
            (v.id.clone(), v.size, v.alloc_loc.clone(), true, v.initial_content.clone(), v.alloc_defined)
        })
        .filter(|v| !v.5)
        .map(|v| (v.0, v.1, v.2, v.3, v.4))
        .collect();

    let deallocations: Vec<_> = alloc_tracker.vars.iter()
        .map(|(_, v)| {
            (v.id.clone(), v.size, v.dealloc_loc.clone(), false, None)
        })
        .filter(|v| v.2.is_some())
        .map(|v| (v.0, v.1, v.2.unwrap(), v.3, v.4)) 
        .collect();

    total_list.extend(deallocations);
    total_list.sort_by(|a, b| a.2.loc.cmp(&b.2.loc));

    let mut delete_counter: HashMap<String, usize> = HashMap::new();

    kernel_proc.apply(&mut |proc| {
        for (var_id, size, loc, is_alloc, content) in total_list.iter() {
            if loc.proc_id == proc.id {
                let r = delete_counter
                    .entry(proc.id.clone()) 
                    .and_modify(|v| *v += 1)
                    .or_insert(0);

                let k = if *is_alloc { Kernels::Alloc { 
                    id: var_id.clone(),
                    size: *size,
                    content: content.clone(),
                } } else { Kernels::Dealloc { 
                    id: var_id.clone(),
                    size: *size 
                } };

                let loc = loc.loc + *r;
                let loc = if loc < (proc.len()-1) && !*is_alloc { loc + 1 } else { loc };

                proc.insert(loc, k)
                
            }
        }
    });
}