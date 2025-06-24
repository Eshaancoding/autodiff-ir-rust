use crate::trackers::KernelTracker;

impl<'a> KernelTracker<'a> {
    pub fn merge (&mut self, other: &'a KernelTracker) {
        /* 
        merge kernel tracker with cloned kernel tracker (this is called during branching) 

        detect any other conflicts!! Gotta check on this
        */
        self.alloc_tracker = other.alloc_tracker.clone();
    }
}