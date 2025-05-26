# Matrix Multiplication Testing

The goal of this folder is to experiment with new matrix multiplication techniques for CPUs in C++

Why custom kernels? We would have to generate this C++ code anyways. It is entirely possible that we can use OpenBLAS to do this, but if we can program custom C++ kernels that performs at around OpenBLAS level anyways, then why try not use these custom kernels.

The added plus with these custom kernels is that it makes kernel fusion not only easier, but also potentially faster as we do not need to reread memory (all of the operations are done when the results are at the registers anyways).

So, it is a pretty much win-win situation here. Plus, I get to learn more about generating efficient code on CPUs (and hopefully transfer some of my knowledge into GPU optimization as well).

## Organization

* ~~matmul.cpp~~: The code for the matrix multiplication (using AVX 256 version to compare with `sgemm.c`)

* ~~512_matmul.cpp~~: The code for the matrix multiplication (using AVX 512 to compare with pytorch & numpy)

* ~~Makefile~~: where you run `matmul.cpp` (`make`) and `512_matmul.cpp` (`make 512`). 

* ~~test_numpy.py~~: The code for testing GFLOPS for numpy

* ~~test_torch.py~~: The code for testing GFLOPS for torch

## Results

The 512 implementation (~200 GFLOPS) gets comparable performance to Numpy dot product (~200 GFLOPS) - Aka a OpenBLAS backend. 256 implementation is not that far off, honestly (~180 GFLOPS). Pytorch is also somewhat slightly ahead with 213 GFLOPS. It's pretty clear that most of the bottleneck is memory accessing (which I have an idea to make this better :D)

However, there's always better implementations. I am comparing myself against `sgemm.c`, which boasts a ~250 - 280 GFLOPS! This is ideally where I want the implementation to go for.

Take this results with a grain of salt however... This is tested on a really shitty tCPU with WSL enabled... a terrible platform to test this. I will test on a better CPU backend with more # of trials in the future.