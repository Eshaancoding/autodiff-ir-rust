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
    CreateConstant {contents: f64, id: String, dim: Vec<usize>},

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

    /*
    Reduce Operations
    All tensors to be reduced MUST be 2-dimensional and must reduce over the last (2nd) dimension

    To handle any dimension of reduction, we just permute:
        torch.all(torch.permute(torch.permute(a, [0,3,2,1]).sum(dim=-1), [0,2,1]) == a.sum(dim=1))

    To handle any # of dimensions, we just reshape (view)
        torch.all(torch.abs(torch.permute(torch.permute(a, [0,3,2,1]).reshape(-1, 16).sum(dim=-1).reshape(8, 256, 32), [0,2,1]) - a.sum(dim=1)) <= 1e-5)

    In summary, the procedure is:
        1. Permute dim to reduce to the last dimension
        2. Reshape to (X,Y) --> (-1, # of elements in dim to reduced)
        3. reduce across last dimension
        4. Reshape to orig dim (but remove reduced dim, since we already did that)
        5. permute back to original.

    We do this because it simplifies the reduce kernel for each device. We take advantage of the fact that data manipulation is 0-cost unless actually needed.
    */
    Sum {a: String, res: String}, 

    // add max, min (min can be in terms of max), etc.
    // add max within the elew kernel

    /* 
    Technically, Dot Product CAN BE expressed with Sum and ElwMult. In fact, this is what TinyGrad does
    However, I recognize that there are very specific, optimized implementations of Dot Product (CUBLAS)
    TinyGrad does recognizes this and applies these specific opts implementations if necessary, but descerning them from list of IR instructions POV is weird.
    Considering its importance in machine learning, I decided to seperate it to a seperate command itself.
    */
    DotProduct {a: String, b: String, res: String},         // (a,b) x (b,c) --> (a,c). Both tensors are 2-dim.

    // Data Manipulation --> every operation is 0-cost except Contigious
    // At kernel level, we just use fancy indexing. Check out matrix tracker (kernel/trackers/matrix.rs) and access expression generation (kernel/access_expr.rs)
    View      {a: String, target_dim: Vec<usize>, res: String},
    Index     {a: String, index: usize, dim: usize, res: String}, 
    Concat    {a: String, b: String, dim: usize, res: String},
    Permute   {a: String, p: Vec<usize>, res: String}, 
    Broadcast {a: String, dim: usize, r: usize, res: String}, 
    Contigious {a: String, res: String},  // all the above operations use fancy indexing. However, this operation constructs the full matrix explicitly.

    // Single-input Functions
    Exp2  {a: String, res: String},   // fine: shift?
    Log2  {a: String, res: String},   // fine: counting?
    Sin   {a: String, res: String},   // not as fine: lookup table + quadratic interpolation (c0 + c1*x + c2*x*x); implement at conditioner
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
    // Execute a list of instructions 
    fn execute (&mut self, cmds: IndexMap<String, Vec<IRCmds>>);
    
    // Transfers matrix id to device.
    fn get_tensor (&self, id: &String) -> ValueData;
    
    // If the device needs any specific requirements / changes to the IR before passing to IR optimization, you can declare it here.
    // If no optimizations needed, then just leave this function empty
    fn ir_callback (&self, cmds: &mut IRBase);

    // If there needs to be any shape override for dot product, you can declare it here.
    // There are many implementations of dot product a device can have (ex: b might need to be transposed for column-wise accessing).
    // Therefore, this function exists to accomodate any DP implementations
    fn dot_prod_shape (&self, a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
        vec![a.first().unwrap().clone(), b.last().unwrap().clone()]
    }
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

pub fn ir_b_id () -> String {
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    let x = ir_b.unique_id();
    drop(guard);
    return x
}

pub fn ir_b_device_callback () {
    let mut guard_device = DEVICE.lock().unwrap();
    let mut guard = IRB.lock().unwrap();

    let device = guard_device.as_mut().expect("Can't unpack device");
    let mut ir_b = guard.as_mut().expect("Can't unpack IRBuilder");

    device.ir_callback(&mut ir_b); 

    drop(guard_device);
    drop(guard);
}

pub fn ir_b_execute () {
    // get cmds
    let mut guard = IRB.lock().unwrap();
    let ir_b = guard.as_mut().expect("Can't unpack IRBuilder");
    let cmds = ir_b.cmds.clone();
    drop(guard);

    let mut guard = DEVICE.lock().unwrap();
    let device = guard.as_mut().expect("Can't unpack IRBuilder");
    device.execute(cmds);
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