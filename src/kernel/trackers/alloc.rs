use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use crate::{ir::helper::{ir_to_res, ir_to_dep}, IRCmds};
use super::ShapeTracker;

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
    
    // If the deallocation location does not exist, then we know it is a dependency variable
    pub dealloc_loc: Option<Location>
}

#[derive(Clone)]
pub struct AllocTracker<'a> {
    // hlir id is same as the alloc id
    // However, not all hlir id will be in vars (ex: referencing source vars)
    pub vars: HashMap<String, AllocEntry>,  
    pub alloc: HashMap<String, Location>,
    pub dealloc: HashMap<String, Location>,
    pub dep_vars: &'a HashSet<String>
}

impl<'a> AllocTracker<'a> {
    pub fn new (dep_vars: &'a HashSet<String>) -> AllocTracker<'a> {
        AllocTracker {  
            vars: HashMap::new(),
            alloc: HashMap::new(),
            dealloc: HashMap::new(),
            dep_vars
        }
    }
    
    pub fn update_vars (&mut self, ir_id: &String, dim: &Vec<usize>, loc: Location) {
        let total_dim = dim.iter().product::<usize>();
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
                    dealloc_loc: None
                }
            );
        }
    }
    
    pub fn step (&mut self, cmd: &'a IRCmds, shape_tracker: &ShapeTracker, loc: Location) {
        // step shape tracker before stepping alloc tracker
        match cmd {
            // ignore all data manipulation cmds
            IRCmds::View { .. } => {}, 
            IRCmds::Index { .. } => {}, 
            IRCmds::Concat { .. } => {}, 
            IRCmds::Permute { .. } => {}, 
            IRCmds::Broadcast { .. } => {}, 
            IRCmds::CreateConstant { .. } => { }, // skip create constant
            IRCmds::CreateMat { contents, dim, id } => {
                self.vars.insert(
                    id.clone(),
                    AllocEntry { 
                        id: id.clone(), 
                        size: dim.iter().product::<usize>(),
                        initial_content: Some(contents.clone()),
                        alloc_loc: 
                            if self.dep_vars.contains(id) {
                                Location {
                                    proc_id: "main".to_string(),
                                    loc: 0
                                }
                            } else {
                                loc.clone()
                            },
                        dealloc_loc: None
                    } 
                );
            },
            _ => {
                if let Some(id) = ir_to_res(cmd) {
                    let sh = shape_tracker.get_shape(&id).clone();
                    self.update_vars(&id, &sh, loc.clone());
                }
            }
        }

        for v in ir_to_dep(cmd) {
            // skip dealloc loc if it contains one of the dep variables
            if self.dep_vars.contains(v) { continue; }
            
            // update dealloc location
            if let Some(entry) = self.vars.get_mut(v) {
                entry.dealloc_loc = Some(loc.clone());
            }
        }

    }  

    pub fn get_alloc (&self, id: &String) -> &AllocEntry {
        self.vars.get(id).unwrap()
    }
}