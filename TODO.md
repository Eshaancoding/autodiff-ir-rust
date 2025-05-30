# TODO
Frontend will refer to the design of the developer functions/ways to interact with the IR. It also includes NN libraries and what not.

Backend will refer to things that runs the internal operations and optimizations of said operations.

## Frontend

* ~~Softmax~~
* ~~ReLU~~
* Different losses: 
    * Cross entropy
    * MSE
    * etc.
* ~~RmsNorm   <-- Needed for Transformer~~
* ~~LayerNorm <-- Needed for Transformer~~
* Dropout   <-- Dropout
* All of the above operations should be enough for building up to attention
* Transformer
* Implement Adam
    * apparently there's a general impl of optimizers that impls Adam and others...
        * check that!
* batch matrix multiplication
    * view with -1
* NN operations:
    * Convolutions
    * ConvTranspose
    * Max/Avg/Fractional Pooling layers <-- create general pooling operator (like op)
    * zero padding --> concat under the hood
    * Activations:
        * ELU
        * GeLU
        * SeLU
    * Batch Norm
    * GNN/Cell,RNN/Cell,etc.
    * More Loss funcs:
        * L1Loss
        * NLLLoss
        * KLDiv loss
    * .product(); like .sum()?

* Other operations?
    * [https://pytorch.org/docs/stable/torch.html](Pytorch)
    * vstack, hstack <-- simple wrapper over cat
    * split tensor
    * tile
* DataLoader (use feeder)
* Simple operations:
    * abs
    * random trig/almost trig funcs: 
        * logs
        * acosh
        * acos
        * inverse tan
        * etc. <-- do this when you know how to function with weird funcs
    * argmax/argmin
    * min/max <-- simple wrapper over idx
    * more random distribution generators
        * berournelli, etc.

## Backend

* ~~1. Remove Neg IR instruction~~

* **Memory**:
    * Get the "special IR function" callback that is customized per device --> then for x86 dot product add transpose before dot production of `A` in `AB` matrix mul

    * automatic allocation and deallocation within graph
        * tracing through BRs will be challenging, however...
        * change kernel_decl.rs and do efficiently

    * ~~1. Remove binary/unary funcs and just use general "Expression"~~

    * <mark>2. Update Matrix Tracker tracker for dot prod kernels</mark>

    * <mark>3. Update matrix tracker for reduce kernel</mark>

    * Finish Concat logic in matrix tracker

    * Finish constant logic in matrix tracker

    * CPU --> GPU feeder 
        * aka dataset

    * **ADVANCED**: Allow dot product and sum and other advance kernels to be accessed WITHOUT CONTIGIOUS
        * This is a problem because dot produce/reduce kernels often have `load` instructions that load 4 bytes + orig
            * sometimes, it just needs to load ONE (think)
        * this needs some kernel variant of some sort, which is really weird. So for now, we have to assume contigious.
        * technically, you only need the weight tensor to be in full contigious, as the value in the x value is just constant
            * look at the implementation :D
            * This is the only case for the CPU. NVIDIA / etc. implementations might be different.
            * per device, you probably need to provide whether the support for certain fusion implementations is common.
                * probably pass it as param to `to_kernel`

* **X86**:
    * Allow dot prod implementation to support varied shapes rather than just power of 2
    * <mark>3. Efficient Reduce kernel.</mark>

* **Kernel experimentation:**
    * Experiment with different parameters of dot prod + other kernels 
        * [this](https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html) does a good job
        * there's other optimizations, I am sure. Don't focus on that right now, have the general base for everything first.

    * Contigious memory vs. direct accessing for dot prod kernels
        * efficient dot product kernels assume that it is contigious 
        * Furthermore, we assume that that `A` in `Ax` in matrix multiplication is **column-wise** rather than **row-wise**
            * need to manually assume that there's a transpose before the A in matrix multiplication.

    * How/where to organize this? Each device will have different kernels which will have different params to opts...
        * probably within each device?  
        * **YOU NEED BOTH**

* **Kernel Fusion**
    * do basic multiple binary/unary kernel fusion
    * dot prod impl is kinda wacky
        * access expression assumes global id...
    * anyway to do dotprod fusion?  
        * similar to flash attention
        * look into optimized [cuda matmul kernel](https://siboehm.com/articles/22/CUDA-MMM)
        * even better optimization for [kernels](https://salykova.github.io/sgemm-gpu)
        * technically, there's even more [kernels at llm.c](https://github.com/karpathy/llm.c/tree/master/dev/cuda)
        * more kernel opt (+ read kernel fusion) [here](https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html)
        * transpose operator faster: [here](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/)
            * prolly uses this: [here](https://veitner.bearblog.dev/tma-introduction/)
        * even faster kernel stuff for generation: [here](https://www.together.ai/blog/chipmunk)
            * The entire purpose of **TogetherAI** is optimizing kernels in a way.
        * There are more and more special features of hardware on more and more GPUs:
            * [link](https://tridao.me/blog/2024/flash3/)
        * life is not all that simple now is it hehe
    * you need more knowledge of all of this before you go into this optimizations
        * not sure if you can beat hand-tune optimizations

* **HLIR Opts**
    * if matrix is always used in it's transposed form, then set the contents such that it is in transposed and remove transpose operation
        * Good for weight optimization :)

    * View removal
        * If multiple views in sequence, just turn it into the one single view (the last view operation)
        * if view is already in shape, then delete

    * Remove double permutations
        * if `b = a.permute([1, 0])` followed by `c = b.permute([1, 0])`, this is the same as the original input `a`
        * transpose via transpose

    * Constant evaluator: including 0 and 1 tracking
        * 0 * val --> 0; optimize
        * ~~1 * val --> val; optimize~~
        * 0 + val --> val; optimize
        * if a = 2, b = 3, and c = a * b, then set to 6

    * More aggressive IR optimizations for localization:
        * Also test for RNN, Transformers --> improve library
        * Then you can improve the IR optimizations as well. 
            * etc. etc. etc. 

    * Not sure if you can improve even further/less bugs if you turn it into a GRAPH rather than a list of optimizations
        * maybe some optimizations can benefit from this, not everything...

        * `to_graph` func should be created and used across IRs that benefit from it.
            * good for debugging as well



* **General Ideas**: 
    * Dynamic Shape

    * Cuda graphs
        * have to use BR compiler hints in order to determine while or if statement
        * you may also need to send extra compiler hints 
        * skdjfksjdfkjsdkfjskdfjksdjf that's also going to be pretty weird.



# Rust Codebase

* Remove excessive clones