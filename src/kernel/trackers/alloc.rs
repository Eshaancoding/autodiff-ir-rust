use std::collections::HashMap;
use crate::{ir::helper::{ir_to_res, ir_to_dep}, IRCmds};
use super::ShapeTracker;

#[derive(Clone, Debug)]
pub struct Location {
    pub proc_id: String,
    pub loc: usize
}

#[derive(Clone, Debug)]
pub struct AllocEntry<'a> {
    pub id: String,
    pub size: usize,
    pub initial_content: Option<&'a Vec<f64>>,
    pub alloc_loc: Location,
    pub dealloc_loc: Location
    // if there's data needed to be allocation before running program, then specify what content this is
    // If this value is none, then some sort of computation needs to be done before this appears.
    // also note that the size of the initial content is not the same as the size of the alloc itself!
}

#[derive(Clone)]
pub struct AllocTracker<'a> {
    // hlir id is same as the alloc id
    // However, not all hlir id will be in vars (ex: referencing source vars)
    pub vars: HashMap<String, AllocEntry<'a>>,  
    pub alloc: HashMap<String, Location>,
    pub dealloc: HashMap<String, Location>
}

impl<'a> AllocTracker<'a> {
    pub fn new () -> AllocTracker<'a> {
        AllocTracker {  
            vars: HashMap::new(),
            alloc: HashMap::new(),
            dealloc: HashMap::new()
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
                    alloc_loc: loc.clone(),
                    dealloc_loc: loc
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
            IRCmds::CreateMat { contents, dim, id } => {
                self.vars.insert(
                    id.clone(),
                    AllocEntry { 
                        id: id.clone(), 
                        size: dim.iter().product::<usize>(),
                        initial_content: Some(contents),
                        alloc_loc: loc.clone(),
                        dealloc_loc: loc.clone()
                    } 
                );
            },
            IRCmds::CreateConstant { .. } => { }, // skip create constant
            _ => {
                if let Some(id) = ir_to_res(cmd) {
                    let sh = shape_tracker.get_shape(&id).clone();
                    self.update_vars(&id, &sh, loc.clone());
                }
            }
        }

        for v in ir_to_dep(cmd) {
            if let Some(entry) = self.vars.get_mut(v) {
                entry.dealloc_loc = loc.clone();
            }
        }

    }  

    pub fn get_alloc (&self, id: &String) -> &AllocEntry {
        self.vars.get(id).unwrap()
    }
}