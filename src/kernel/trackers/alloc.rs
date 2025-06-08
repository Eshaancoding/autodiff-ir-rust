use std::collections::HashMap;
use crate::{ir::optimizations::helper::ir_to_res, Device, IRCmds};
use super::ShapeTracker;

#[derive(Clone, Debug)]
pub struct AllocEntry<'a> {
    pub id: String,
    pub size: usize,
    pub initial_content: Option<&'a Vec<f64>> 
    // if there's data needed to be allocation before running program, then specify what content this is
    // If this value is none, then some sort of computation needs to be done before this appears.
    // also note that the size of the initial content is not the same as the size of the alloc itself!
}
pub struct AllocTracker<'a> {
    shape_tracker: ShapeTracker,
    device: &'a dyn Device,

    // hlir id is same as the alloc id
    // However, not all hlir id will be in vars (ex: referencing source vars)
    pub vars: HashMap<String, AllocEntry<'a>>,  

}

impl<'a> AllocTracker<'a> {
    pub fn new (device: &'a dyn Device) -> AllocTracker<'a> {
        AllocTracker {  
            device,
            vars: HashMap::new(),
            shape_tracker: ShapeTracker::new()
        }
    }
    
    pub fn update_vars (&mut self, ir_id: &String, dim: &Vec<usize>) {
        let total_dim = dim.iter().product::<usize>();
        if self.vars.contains_key(ir_id) {
            self.vars.entry(ir_id.clone()).and_modify(|f| {
                f.size = f.size.max(total_dim)
            });
        } else {
            self.vars.insert(
                ir_id.clone(), 
                AllocEntry { id: ir_id.clone(), size: total_dim, initial_content: None }
            );
        }
    }
    
    pub fn step (&mut self, cmd: &'a IRCmds) {
        self.shape_tracker.step(self.device, cmd);
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
                        initial_content: Some(contents)
                    } 
                );
            },
            _ => {
                if let Some(id) = ir_to_res(cmd.clone()) {
                    let sh = self.shape_tracker.get_shape(&id).clone();
                    self.update_vars(&id, &sh);
                }
            }
        }
    }  

    pub fn get_alloc (&self, id: &String) -> &AllocEntry {
        self.vars.get(id).unwrap()
    }
}