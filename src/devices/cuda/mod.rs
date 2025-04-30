/*
to be created
you can use cudarc for launching kernels + abstracting over cuBLAS

There are many optimizations for kernel generation. TinyGrad does a very interesting job at this: 
https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html
https://www.abhik.xyz/articles/compiling-pytorch-kernel <-- this might also be helpful on PyTorch optimization

Since the architecture of CUDA is inherently "thread-like", you can go into a lot of kernel optimization. 
Example:
* How much computation is each thread allowed to compute (1 thread responsible for two element wise operations)
    * block dim, grid dim, etc.
* memory allocations (consider CUDA's cache and take advantage of memory localization)
* etc., etc., etc.,

So, CUDA IR Optimization in of itself is a complicated endeavor can take a solid 2-3 weeks of coding if I am 100% productive
*/