use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use opencl3::device::{
    get_all_devices, 
    Device as CLDevice, 
    CL_DEVICE_TYPE_ACCELERATOR, 
    CL_DEVICE_TYPE_ALL, 
    CL_DEVICE_TYPE_CPU, 
    CL_DEVICE_TYPE_GPU
};
use crate::core::ret_dep_list;
use crate::devices::alloc::execute_alloc;
use crate::devices::binary::execute_binary;
use crate::devices::context::OpenCLContext;
use crate::devices::dotprod::execute_dot_prod;
use crate::devices::fuse_dp_elw::execute_fuse_dp_elw;
use crate::devices::fuse_elw::execute_elw_expr;
use crate::devices::fuse_reduce_elw::execute_fuse_reduce_elw;
use crate::devices::movement::execute_movement;
use crate::devices::reduce::execute_reduce;
use crate::devices::unary::execute_unary;
use crate::kernel_decl::{KernelProcedure, Kernels};
use crate::trackers::KernelTracker;
use crate::{IRBase, ValueData, Device};

pub enum CLDeviceType {
    CPU,
    GPU,
    ACCELERATOR,
    ALL
}

pub struct OpenCL {
    device: CLDevice,
    result: HashMap<String, Arc<Vec<f32>>>,
    result_shape: HashMap<String, Vec<usize>>
}

impl OpenCL {
    pub fn new (device_type: CLDeviceType) -> OpenCL {
        let device = CLDevice::new(
            *get_all_devices(match device_type {
                CLDeviceType::ALL => { CL_DEVICE_TYPE_ALL },
                CLDeviceType::CPU => { CL_DEVICE_TYPE_CPU },
                CLDeviceType::GPU => { CL_DEVICE_TYPE_GPU }
                CLDeviceType::ACCELERATOR => { CL_DEVICE_TYPE_ACCELERATOR }
            }).expect("Unable to get all devices").first().expect("No device found!")
        );

        println!("Using device: {}", device.name().expect("Can't get device name"));

        OpenCL { 
            device,
            result: HashMap::new(),
            result_shape: HashMap::new()
        }
    }
}

impl Device for OpenCL {
    fn execute (&mut self, proc: KernelProcedure, tracker: KernelTracker) {
        self.result.clear();
        self.result_shape.clear();
        let mut context = OpenCLContext::new(self.device);
        
        // warmup; compile programs, initialize buffers + writing to those buffers, executing kernels, etc.
        println!("Compiling...");
        let mut proc = proc;
        proc.step_cmd(&mut |v, idx| {
            let cmd = v.get(*idx).unwrap();
            exec(cmd, &mut context); 
            true
        });

        // Actually Execute
        println!("Executing...");
        let start = Instant::now();
        proc_exec(&proc, &mut context);
        println!("elapsed: {} s", start.elapsed().as_secs_f64());

        // From all dep list, get variables
        let dep_list = ret_dep_list();
        for st in dep_list.iter() {
            self.result.insert(st.clone(), Arc::new(context.read_buffer(st)));
            self.result_shape.insert(st.clone(), tracker.get_shape(st).clone());
        }
    }

    fn get_tensor (&self, id: &String) -> ValueData {
        // not implemented yet
        if let Some(data) = self.result.get(id) {
            ValueData {
                id: id.clone(),
                dim: self.result_shape.get(id).unwrap().clone(),
                data: data.clone(), 
                is_none: false
            }
        } else {
            ValueData::none()  
        }
    }

    fn ir_callback (&self, _: &mut IRBase) {}
}

fn proc_exec (proc: &KernelProcedure, opencl_context: &mut OpenCLContext) -> bool {
    let mut exit = false;    

    // there's no If/While constructs in OpenCL (there IS in CUDA)
    // so, we run stuff in the host side of things.
    // This pretty much ruins the point of control statement in the OpenCL side
    // But, it's nice to have in other backends
    for cmd in proc.iter() {
        if let Kernels::EX {} = cmd {
            return true;
        }
        else if let Kernels::If { conditions, else_proc } = cmd {
            let mut run_cond = false;
            for (cond, c_proc) in conditions.iter() {
                if opencl_context.read_buffer(cond)[0] == 1.0 {
                    exit = proc_exec(c_proc, opencl_context); // run whatever is inside condition
                    run_cond = true;               // set run condition
                    break;                         // don't eval any other conditions
                }
            }        

            if let Some(e_proc) = else_proc {
                if !run_cond { exit = proc_exec(e_proc, opencl_context); } 
            }
        }
        else if let Kernels::While { conditional_var, block } = cmd {
            while opencl_context.read_buffer(conditional_var)[0] != 0.0 {
                exit = proc_exec(block, opencl_context);
                if exit { break; }
            }
        }
        else {
            exec(cmd, opencl_context);            
        }

        if exit { return true; }
    }

    false
}

fn exec (cmd: &Kernels, opencl_context: &mut OpenCLContext) {
    match cmd {
        Kernels::Alloc { .. } => { execute_alloc(opencl_context, cmd); },
        Kernels::Dealloc { .. } => { execute_alloc(opencl_context, cmd); },
        Kernels::Unary { .. } => { execute_unary(opencl_context, cmd); },
        Kernels::Binary { .. } => { execute_binary(opencl_context, cmd); },
        Kernels::DotProd { .. } => { execute_dot_prod(opencl_context, cmd); },
        Kernels::Reduce { .. } => { execute_reduce(opencl_context, cmd); },
        Kernels::Movement { .. } => { execute_movement(opencl_context, cmd); },
        Kernels::ElwExpr { .. } => { execute_elw_expr(opencl_context, cmd); } ,
        Kernels::DPElwExpr { .. } => { execute_fuse_dp_elw(opencl_context, cmd); },
        Kernels::ReduceElwExpr { .. } => { execute_fuse_reduce_elw(opencl_context, cmd); },
        Kernels::While { .. } => {}, // handled by parent funcs
        Kernels::If { .. } => {}, // handled by parent funcs
        Kernels::EX { .. } => {}, // handled by parent funcs
    }
}