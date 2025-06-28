use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use crate::kernel_decl::{Kernels};
use crate::trackers::ShapeTrackerKernel;
use crate::Device;

#[derive(Clone, Debug)]
pub struct Location {
    pub proc_id: String,
    pub loc: usize
}

#[derive(Clone, Debug)]
pub struct AllocEntry {
    pub id: String,
    pub size: usize,

    // if there's data needed to be allocation before running program, then specify what content this is
    // If this value is none, then some sort of computation needs to be done before this appears.
    // also note that the size of the initial content is not the same as the size of the alloc itself!
    pub initial_content: Option<Arc<Vec<f64>>>,

    pub alloc_loc: Location,
    pub alloc_defined: bool,  // Alloc is already defined
    
    // If the deallocation location does not exist, then we know it is a dependency variable
    pub dealloc_loc: Option<Location>
}

#[derive(Clone)]
pub struct AllocTracker<'a> {
    // hlir id is same as the alloc id
    // However, not all hlir id will be in vars (ex: referencing source vars)
    pub vars: HashMap<String, AllocEntry>,  
    pub dep_vars: &'a HashSet<String>,
    pub shape_tracker: ShapeTrackerKernel
}

impl<'a> AllocTracker<'a> {
    pub fn new (dep_vars: &'a HashSet<String>) -> AllocTracker<'a> {
        AllocTracker {  
            vars: HashMap::new(),
            dep_vars,
            shape_tracker: ShapeTrackerKernel::new()
        }
    }
    
    pub fn update_vars (&mut self, ir_id: &String, total_dim: usize, loc: Location) {
        if self.vars.contains_key(ir_id) {
            self.vars.entry(ir_id.clone()).and_modify(|f| {
                f.size = f.size.max(total_dim)
            });
        } else {
            self.vars.insert(
                ir_id.clone(), 
                AllocEntry { 
                    id: ir_id.clone(), 
                    size: total_dim, 
                    initial_content: None,
                    alloc_loc: 
                        if self.dep_vars.contains(ir_id) {
                            Location {
                                proc_id: "main".to_string(),
                                loc: 0
                            }
                        } else {
                            loc.clone()
                        },
                    alloc_defined: false,
                    dealloc_loc: None
                }
            );
        }
    }
    
    pub fn step (&mut self, device: &dyn Device, cmd: &'a Kernels, loc: Location) {
        self.shape_tracker.step(device, cmd);

        match cmd {
            // ignore all data manipulation cmds
            Kernels::Alloc { content, size, id, .. } => {
                self.vars.insert(
                    id.clone(),
                    AllocEntry { 
                        id: id.clone(), 
                        size: *size,
                        initial_content: content.clone(),
                        alloc_loc: loc.clone(), 
                        alloc_defined: true,
                        dealloc_loc: None
                    } 
                );
            },
            _ => {
                if let Some(id) = cmd.get_res() {
                    let sh = self.shape_tracker.get_shape(&id).clone();
                    self.update_vars(&id, sh, loc.clone());
                }
            }
        }

        for v in cmd.get_dep_id() {
            // skip dealloc loc if it contains one of the dep variables
            if self.dep_vars.contains(v) { continue; }
            
            // update dealloc location
            if let Some(entry) = self.vars.get_mut(v) {
                entry.dealloc_loc = Some(loc.clone());
            }
        }

    }  

    pub fn merge (&mut self, other: AllocTracker, merge_loc: &Location) {
        self.vars = other.vars;    
        self.shape_tracker = other.shape_tracker;

        for (_, entry) in self.vars.iter_mut() {
            if let Some(dealloc_loc) = entry.dealloc_loc.as_mut() {
                if dealloc_loc.proc_id != entry.alloc_loc.proc_id {
                    *dealloc_loc = merge_loc.clone();
                }
            }
        }
    }

    pub fn get_alloc (&self, id: &String) -> &AllocEntry {
        self.vars.get(id).unwrap()
    }
}