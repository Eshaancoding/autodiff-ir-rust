pub mod insert_alloc;
pub mod alloc_switch;
pub mod alloc_temp_opt;
pub mod alloc_out_fused;
pub mod alloc_in;
pub mod tetris_opt;

pub use insert_alloc::*;
pub use alloc_switch::*;
pub use alloc_temp_opt::*;
pub use alloc_out_fused::*;
pub use tetris_opt::*;
pub use alloc_in::*;