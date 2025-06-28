use crate::trackers::{KernelTracker, Location};

impl KernelTracker {
    pub fn merge (&mut self, _other: KernelTracker, _merge_loc: &Location) {
        /* 
        merge kernel tracker with cloned kernel tracker (this is called during branching) 

        detect any other conflicts!! Gotta check on this
        */

        // ======================== Allocation ======================== 
        // detect if any variables that have the allocation that is not in the same proc id
        
        /*
        self.alloc_tracker = other.alloc_tracker.clone();
        for (_, entry) in self.alloc_tracker.vars.iter_mut() {
            if let Some(dealloc_loc) = entry.dealloc_loc.as_mut() {
                if dealloc_loc.proc_id != entry.alloc_loc.proc_id {
                    *dealloc_loc = merge_loc.clone();
                }
            }
        }
        */
    }
}