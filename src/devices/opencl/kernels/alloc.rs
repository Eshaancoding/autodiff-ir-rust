use crate::{devices::context::OpenCLContext, kernel_decl::Kernels};

pub fn execute_alloc (opencl_context: &mut OpenCLContext, cmd: &Kernels) {
    match cmd {
        Kernels::Alloc { id, size, content } => {
            if let Some(c) = content {
                assert_eq!(c.len(), *size, "Size of content is not equal size of alloc (opencl not supported)");
                opencl_context.write_buffer(id, c);
            }
            else {
                opencl_context.create_buffer(id, *size);
            }
        },
        Kernels::Dealloc { .. } => {
            // we actually don't dealloc. All buffers will be dropped as soon as the program ends thanks to opencl3
            // Even if there's an Alloc (with a write) and a dealloc inside a while loop
            // we will keep the buffer of dealloc. However, during the alloc, we will just write to the already existing created buffer
        },
        _ => {}
    }
}