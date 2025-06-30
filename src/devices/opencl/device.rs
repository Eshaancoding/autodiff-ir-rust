use opencl3::device::{
    get_all_devices, 
    Device as CLDevice, 
    CL_DEVICE_TYPE_ACCELERATOR, 
    CL_DEVICE_TYPE_ALL, 
    CL_DEVICE_TYPE_CPU, 
    CL_DEVICE_TYPE_GPU
};

use opencl3::context::Context;
use crate::to_kernel::to_kernel;
use crate::{Device, IRBase, IRProcedure, ValueData};

pub enum CLDeviceType {
    CPU,
    GPU,
    ACCELERATOR,
    ALL
}

pub struct OpenCL {
    device: CLDevice,
    context: Context
}

impl OpenCL {
    fn run_alloc () {}
    fn run_dealloc () {} 
    fn run_kernel () {}

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

        let context = Context::from_device(&device).expect("Can't create context from device");

        OpenCL { 
            device,
            context
        }
    }
}

impl Device for OpenCL {
    fn execute (&mut self, proc: &IRProcedure) {
        let kernel_procedure = to_kernel(self, proc);
    }

    fn get_tensor (&self, _: &String) -> ValueData {
        // not implemented yet
        ValueData::none()  
    }

    fn ir_callback (&self, _: &mut IRBase) {}
}