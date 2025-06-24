// insert allocations and deallocations accordant to alloc tracker
use std::collections::HashMap;
use crate::{kernel_decl::{KernelProcedure, Kernels}, trackers::AllocTracker};

pub fn insert_alloc<'a> (kernel_proc: &mut KernelProcedure, alloc_tracker: &AllocTracker<'a>) {
    let mut total_list: Vec<_> = alloc_tracker.vars.iter().map(|(_, v)| {
        (v.id.clone(), v.size, v.alloc_loc.clone(), true)
    }).collect();

    let deallocations: Vec<_> = alloc_tracker.vars.iter().map(|(_, v)| {
        (v.id.clone(), v.size, v.dealloc_loc.clone(), false)
    }).collect();

    total_list.extend(deallocations);
    total_list.sort_by(|a, b| a.2.loc.cmp(&b.2.loc));

    let mut delete_counter: HashMap<String, usize> = HashMap::new();

    kernel_proc.apply(&mut |proc| {
        for (var_id, size, loc, is_alloc) in total_list.iter() {
            if loc.proc_id == proc.id {
                let r = delete_counter
                    .entry(proc.id.clone()) 
                    .and_modify(|v| *v += 1)
                    .or_insert(0);

                let k = if *is_alloc { Kernels::Alloc { 
                    id: var_id.clone(),
                    size: *size 
                } } else { Kernels::Dealloc { 
                    id: var_id.clone(),
                    size: *size 
                } };
                proc.insert(loc.loc + *r, k)
            }
        }
    });
}