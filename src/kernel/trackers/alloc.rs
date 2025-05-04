use std::collections::HashMap;

use crate::{ir::optimizations::helper::ir_to_res, IRCmds};

use super::ShapeTracker;

#[derive(Clone, Debug)]
pub struct AllocEntry<'a> {
    pub id: String,
    pub size: usize,
    pub initial_content: Option<&'a Vec<f64>> 
    // if there's data needed to be allocation before running program, then specify what content this is
    // If this value is none, then some sort of computation needs to be done before this appears.
    // note that even if this alloc is changed in the program, the initial content will stay the same
}
pub struct AllocTracker<'a> {
    id_var: u32,
    shape_tracker: ShapeTracker,
    vars: HashMap<String, AllocEntry<'a>>,  // ir id --> Var Entry
    // note that var entry id IS NOT THE SAME as IR id
}

impl<'a> AllocTracker<'a> {
    pub fn new () -> AllocTracker<'a> {
        AllocTracker {  
            id_var: 0,
            vars: HashMap::new(),
            shape_tracker: ShapeTracker::new()
        }
    }
    
    // same implementation as ir base
    pub fn unique_id (&mut self) -> String {
        let alphabet = "abcdefghijklmnopqrstuvwxyz";
        let base = alphabet.len() as u32;
        let mut idx = self.id_var;
        let mut result = String::new();

        while idx >= base {
            result.push(alphabet.chars().nth((idx % base) as usize).unwrap());
            idx /= base;
            idx -= 1; 
        }
        result.push(alphabet.chars().nth(idx as usize).unwrap());
        self.id_var += 1;

        result.chars().rev().collect() 
    }

    pub fn update_vars (&mut self, ir_id: &String, dim: &Vec<usize>) {
        let total_dim = dim.iter().product::<usize>();
        if self.vars.contains_key(ir_id) {
            self.vars.entry(ir_id.clone()).and_modify(|f| {
                f.size = f.size.max(total_dim)
            });
        } else {
            let uid = self.unique_id();
            self.vars.insert(ir_id.clone(), AllocEntry { id: uid, size: total_dim, initial_content: None });
        }
    }
    
    pub fn step (&mut self, cmd: &'a IRCmds) {
        self.shape_tracker.step(cmd);
        match cmd {
            // ignore all data manipulation cmds
            IRCmds::View { .. } => {}, 
            IRCmds::Index { .. } => {}, 
            IRCmds::Concat { .. } => {}, 
            IRCmds::Permute { .. } => {}, 
            IRCmds::Broadcast { .. } => {}, 
            IRCmds::CreateMat { contents, dim, id } => {
                let uid = self.unique_id();
                self.vars.insert(
                    id.clone(),
                    AllocEntry { 
                        id: uid, 
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
    
    pub fn print (&self) {
        println!("{:#?}", self.vars);
    }
    
}