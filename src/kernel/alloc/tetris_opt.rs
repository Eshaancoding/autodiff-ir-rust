use std::collections::{HashMap, HashSet};

use crate::{core::ret_dep_list, kernel_decl::{Expression, KernelProcedure, Kernels, Matrix}};

// honestly, you need to fix this and metainfo
impl Kernels {
}

#[derive(Debug, Clone)]
struct AllocEntry {
    pub id: String,
    pub start_loc: usize,
    pub end_loc: usize,
    pub size: usize,
    pub offset: usize
}

fn overlap_locs (entry_one: &AllocEntry, entry_two: &AllocEntry) -> bool {
    let start = entry_one.start_loc.max(entry_two.start_loc);
    let end = entry_one.end_loc.min(entry_two.end_loc);

    start < end   
}

pub fn max (a: usize, b: usize) -> usize {
    if b > a { b } else { a }
}

pub fn tetris_opt (kernel_proc: &mut KernelProcedure) {
    let list = ret_dep_list();
    let mut entries: HashMap<String, AllocEntry> = HashMap::new();

    // ===================== Track entries ===================== 
    let mut loc = 0;
    kernel_proc.step_cmd_fusion(&mut |proc, idx| {
        let cmd = proc.get(*idx).unwrap();

        if let Kernels::Alloc { id, size, content } = cmd {
            if content.is_none() && !list.contains(id) {
                entries.insert(id.clone(), AllocEntry {
                    id: id.clone(),
                    start_loc: loc,
                    end_loc: loc,
                    size: *size,
                    offset: 0
                });
            }
        }

        if let Kernels::Dealloc { id, .. } = cmd {
            entries.entry(id.clone())
                .and_modify(|v| v.end_loc = loc);
        }

        loc += 1;
    });

    // ======================== "Tetris" optimization; find offset  ======================== 
    // first sort by size
    let mut entries: Vec<AllocEntry> = entries.iter().map(|v| v.1.clone()).collect();
    entries.sort_by(|a, b| (b.size).cmp(&a.size)); // experiment with this
    
    // then, try to calculate offset
    let mut max_temp_size = 0;
    let mut offsetted_entries: Vec<AllocEntry> = vec![];
    for entry in entries {
        
        let mut offset = 0;
        for offset_entry in offsetted_entries.iter() {
            if overlap_locs(&entry, offset_entry) {
                offset += offset_entry.size;
            }
        }

        let mut new_entry = entry.clone();
        new_entry.offset = offset;
        offsetted_entries.push(new_entry);

        max_temp_size = max(max_temp_size, offset + entry.size)
    }

    println!("Offsetted entries: {:#?}", offsetted_entries);
    println!("max temp size: {}", max_temp_size);

    // ======================== Then, change references and ids ======================== 
    kernel_proc.step_cmd_fusion(&mut |proc, i| {
        let current_kernel = proc.get_mut(*i).unwrap();

        let change_mat = |m: &mut Matrix| {
            for entries in offsetted_entries.iter() {
                if *entries.id == *m.id {
                    if entries.offset > 0 {
                        let new_expr = Expression::make_add(m.access.clone(), Expression::make_const(entries.offset as i32));
                        m.access = new_expr; 
                    }
                    m.id = "_temp".to_string()
                }
            }
        };

        // ideally... move this to helper
        if let Kernels::Unary { a, res, .. } = current_kernel {
            if let Some(m) = a.get_mut_mat() { change_mat(m) }
            if let Some(m) = res.get_mut_mat() { change_mat(m) }
        }
        else if let Kernels::Reduce { a, res, .. } = current_kernel {
            if let Some(m) = a.get_mut_mat() { change_mat(m) }
            if let Some(m) = res.get_mut_mat() { change_mat(m) }
        }
        else if let Kernels::Movement { a, res, .. } = current_kernel {
            if let Some(m) = a.get_mut_mat() { change_mat(m) }
            if let Some(m) = res.get_mut_mat() { change_mat(m) }
        }
        else if let Kernels::Binary { a, b, res, .. } = current_kernel {
            if let Some(m) = a.get_mut_mat() { change_mat(m) }
            if let Some(m) = b.get_mut_mat() { change_mat(m) }
            if let Some(m) = res.get_mut_mat() { change_mat(m) }
        }
        else if let Kernels::DotProd { a, b, res, .. } = current_kernel {
            if let Some(m) = a.get_mut_mat() { change_mat(m) }
            if let Some(m) = b.get_mut_mat() { change_mat(m) }
            if let Some(m) = res.get_mut_mat() { change_mat(m) }
        }
    });

    // ======================== Remove all allocs and deallocs assoacited with the entries ======================== 
    let ids: Vec<String> = offsetted_entries.iter().map(|v| v.id.clone()).collect();
    let mut id_proc = 0;
    let mut did_filter: HashSet<i32> = HashSet::new();

    kernel_proc.step_cmd_fusion(&mut |proc, id| {

        if *id == 0 { id_proc += 1; }        

        if !did_filter.contains(&id_proc) {
            let t: Vec<_> = proc.iter()
                .filter(|p| {
                    if let Kernels::Alloc { id, .. } = p {
                        if ids.contains(id) { return false }
                    }
                    else if let Kernels::Dealloc { id, .. } = p {
                        if ids.contains(id) { return false }
                    } 

                    true
                })
                .map(|v| v.clone())
                .collect();

            *proc = t;
            
            did_filter.insert(id_proc);
        }
        
    });
    
}