use std::string::String;
use indexmap::IndexMap;

use crate::ValueData;
use std::sync::Mutex;

// All different IR needs to implement these functions
// See Tensor_rs
#[derive(PartialEq, Clone, Debug)]
pub enum IRCmds {
    // Create
    CreateMat {contents: Vec<f64>, dim: Vec<usize>, id: String},

    // ELW Operations 
    // No subtraction as a - b = a + (-b)
    // No division as a/b = a * (1/b). (1/b) is using reciprocal.
    // most hardware accel use these optimizations under the hood.
    ElwMultiply   {a: String, b: String, res: String}, 
    ElwAdd        {a: String, b: String, res: String},

    // Operation Equal operations; s -- self, o -- other
    ElwMultiplyEq {s: String, o: String}, // *= 
    ElwAddEq      {s: String, o: String}, // += 

    // Element-wise Equality Operations. Expects tensor with either 0 (doesn't satisfy condition) and 1 (satisfy condition)
    EqualZero {a: String, res: String}, // a == 0 --> res; 
    MoreZero  {a: String, res: String}, // a > 0 --> res
    LessZero  {a: String, res: String}, // a < 0 --> res

    // Reduce operations
    Sum {a: String, dim: usize, res: String}, // different than ElwSum (two matrixes); this all from one matrix
    // add max, min (min can be in terms of max), etc.
    // add max within the elew kernel

    /* 
    Technically, Dot Product CAN BE expressed with Sum and ElwMult. In fact, this is what TinyGrad does
    However, I recognize that there are very specific, optimized implementations of Dot Product (CUBLAS)
    TinyGrad does recognizes this and applies these specific opts implementations if necessary, but descerning them from list of IR instructions POV is weird.
    Considering its importance in machine learning, I decided to seperate it to a seperate command itself.
    */
    DotProduct {a: String, b: String, res: String},         // (a,b) x (b,c) --> (a,c). Both tensors are 2-dim.

    // Data Manipulation --> every operation is 0-cost (unless optimization decides otherwise)
    // At kernel level, we just use indexing maths
    View      {a: String, target_dim: Vec<usize>, res: String},
    Index     {a: String, index: usize, dim: usize, res: String}, 
    Concat    {a: String, b: String, dim: usize, res: String},
    Permute   {a: String, p: Vec<usize>, res: String}, 
    Broadcast {a: String, dim: usize, r: usize, res: String}, 

    // Single-input Functions
    Exp2  {a: String, res: String},   // fine: shift?
    Log2  {a: String, res: String},   // fine: counting?
    Sin   {a: String, res: String},   // not as fine: lookup table + quadratic interpolation (c0 + c1*x + c2*x*x); implement at conditioner
    Neg   {a: String, res: String},   // fine: bit manipulation
    Recip {a: String, res: String},   // lookup-table-driven approximations combined with iterative refinement, optimized for parallel execution and hardware efficiency
    Sqrt  {a: String, res: String},   // calculates inverse square root and then reciprocal...

    // Control Functions
    BR {block_id: String},
    BRE {block_id: String, a: String}, // if a == 1, then branch; if not go to the next cmd
    BRZ {block_id: String, a: String}, // if a == 0, then branch; if not go to the next cmd
    EX,

    // Debug (does not get executed in the slightest)
    Heading {cmt: String},                       // just a comment
    Subheading {h: Option<String>, cmt: String}, // comment that will only appear if at heading (if None, then displays everywhere)
}

#[derive(Clone)]
pub struct IRBase {
    pub id: u32,
    pub current_block: String,
    pub main_block: String,
    pub cmds: IndexMap<String, Vec<IRCmds>>
}

pub static DEVICE: Mutex<Option<Box<dyn Device + Send + Sync>>> = Mutex::new(None);
pub static IRB: Mutex<Option<IRBase>> = Mutex::new(None);

pub trait Device {
    fn execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>);
    fn get_tensor (&self, id: &String) -> ValueData;
}

// helper functions for generating IR
pub fn if_b_id () -> String {
    let mut guard = IRB.lock().unwrap();
    let irb = guard.as_mut().expect("Can't unpack guard");
    irb.unique_id()
}

pub fn ir_b_add (cmd : IRCmds) {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    ir_b.add_cmd(cmd);
    drop(guard);
}

// helper functions for generating IR
pub fn ir_b_id () -> String {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    let x = ir_b.unique_id();
    drop(guard);
    return x
}

pub fn ir_b_execute () {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    let cmds = ir_b.cmds.clone();
    drop(guard);

    let mut guard = DEVICE.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    ir_b.execute(cmds);
    drop(guard);
}

pub fn ir_b_create_block (id: String) {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    ir_b.create_block(id);
    drop(guard);
}

pub fn ir_b_set_main_block (id: String) {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    ir_b.set_main_block(id);
    drop(guard);
}

pub fn ir_b_main_block () {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    ir_b.main_block();
    drop(guard);
}