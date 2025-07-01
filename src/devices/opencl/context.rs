use std::fmt::Display;
use std::{collections::HashMap, ptr::null_mut};
use std::sync::Arc;

use opencl3::kernel::ExecuteKernel;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, 
    device::Device as CLDevice, 
    kernel::Kernel, 
    memory::{Buffer, CL_MEM_READ_WRITE}, 
    program::Program, 
    types::{cl_float, CL_BLOCKING}
};


// Wrapper over the actual opencl3 context, but also includes any variables or compiled programs
// As we go through the program, it will cache any buffers and program compiled
pub struct OpenCLContext {
    pub queue: CommandQueue,
    context: Context,
    buffers: HashMap<String, Buffer<f32>>,
    buffer_size: HashMap<String, usize>,
    kernels: HashMap<String, Kernel>,
    kernel_src: HashMap<String, String>
}

impl OpenCLContext {
    pub fn new (device: CLDevice) -> OpenCLContext {
        let context = Context::from_device(&device)
            .expect("Can't create context from device");

        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("Can't create command queue");
        
        OpenCLContext { 
            context, 
            queue,
            buffers: HashMap::new(),
            buffer_size: HashMap::new(),
            kernels: HashMap::new(),
            kernel_src: HashMap::new()
        }
    }

    pub fn create_buffer (&mut self, id: &String, size: usize) {
        self.write_buffer(id, &Arc::new(vec![0.0; size]));
    }

    pub fn write_buffer (&mut self, id: &String, data: &Arc<Vec<f32>>) {
        let size = data.len();

        let OpenCLContext { queue, buffers, .. } = self;

        let buf = buffers.entry(id.clone()).or_insert(
    unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, size, null_mut()).expect("Can't create buffer")
            }
        );

        self.buffer_size.entry(id.clone())
            .and_modify(|v| assert_eq!(*v, size, "Write size is not equal to alloc size!"))
            .or_insert(size);
        
        let write_event = unsafe {
            queue.enqueue_write_buffer(buf, CL_BLOCKING, 0, &data, &[])
        };

        write_event.expect("Can't create write event")
            .wait()
            .expect("Can't wait for write buffer")
    }

    pub fn read_buffer (&mut self, id: &String) -> Vec<f32> {
        let buf = self.buffers.get(id).expect("Invalid buffer id at reading");
        let size = self.buffer_size.get(id).expect("Invalid buffer id at reading size");

        let mut results: Vec<cl_float> = vec![-1.0; *size];

        let read_event = unsafe {
            self.queue.enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut results, &[])
                .expect("Can't enqueue read buffer")
        };

        read_event.wait().expect("Can't wait for reading buffer");

        results
    }

    pub fn get_kernel<F> (&mut self, kernel_name: &String, gen_src_code: F) -> (&HashMap<String, Buffer<f32>>, ExecuteKernel, &CommandQueue)
        where F: Fn() -> String 
    {
        let OpenCLContext { kernels, context, buffers, queue, .. } = self;

        let k = kernels.entry(kernel_name.clone())
            .or_insert_with(|| {
                let src_code = gen_src_code();
                let program = Program::create_and_build_from_source(&context, &src_code, "")
                    .expect(format!("Can't build program:\n{}", src_code).as_str());
                self.kernel_src.insert(kernel_name.clone(), src_code);

                Kernel::create(&program, kernel_name).expect("Can't create kernel")
            });

        (buffers, ExecuteKernel::new(k), queue)
    }
}

impl Display for OpenCLContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (id, src) in self.kernel_src.iter() {
            let _ = write!(f, "========== For kernel {} ==========\n", id);
            let _ = write!(f, "{}\n\n", src);
        }

        write!(f, "")
    }
}