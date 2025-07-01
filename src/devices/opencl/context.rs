use std::{collections::HashMap, ptr::null_mut};

use opencl3::{command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, context::Context, device::Device as CLDevice, kernel::ExecuteKernel, memory::{Buffer, ClMem, CL_MEM_READ_WRITE}, types::{cl_float, CL_BLOCKING}};


// Wrapper over the actual opencl3 context, but also includes any variables or compiled programs
// As we go through the program, it will cache any buffers or program compiled
pub struct OpenCLContext<'a> {
    context: Context,
    queue: CommandQueue,
    buffers: HashMap<String, Buffer<f32>>,
    buffer_size: HashMap<String, usize>,
    kernels: HashMap<usize, ExecuteKernel<'a>>
}

impl<'a> OpenCLContext<'a> {
    pub fn new (device: CLDevice) -> OpenCLContext<'a> {
        let context = Context::from_device(&device)
            .expect("Can't create context from device");

        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("Can't create command queue");
        
        OpenCLContext { 
            context, 
            queue,
            buffers: HashMap::new(),
            buffer_size: HashMap::new(),
            kernels: HashMap::new()
        }
    }

    pub fn create_buffer (&mut self, id: &String, size: usize) -> &mut Buffer<f32> {
        let res = self.buffers.entry(id.clone()).or_insert(
    unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, size, null_mut()).expect("Can't create buffer")
            }
        );

        self.buffer_size.entry(id.clone()).or_insert(size);

        res
    }

    pub fn write_buffer (&mut self, id: &String, data: &[cl_float]) {
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
}