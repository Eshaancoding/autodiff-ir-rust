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
    * ~~view with -1~~
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
    
* **OpenCL Device**
    * The lowest level!
    * Generate opencl code and stuff
    * with allocations and deallocations
    * ain't that nice.

* fix this 
    * IR simplification (graph)
        * double check the end of the todo and see
        * you should be able to unignore view then

    * support concat of like concat(a, a) <-- already supported?

    * concatenation at the OpenCL level
        * test the evrty.rs
    
    * range select at the OpenCL + kernel level

    * better expression simplification
    
    * fix to_kernel

    * simplify the to_kernel side of things a little bit 
        * still pretty messy, especially the metadata part

* >>>>>> only then you can jump ahead to the multihead attention kernel <<<<<<<

* **Kernel**:
    * ~~Resolve conflicts if memory location of res and memory location of dep ARE EQUAL (same id) and the access expressions are DIFFERENT~~
        * ~~Then, you need a temp allocation for this.~~

    * ~~MEM OPTS Technically, all memory can just be under one HUGE memory block~~
        * ~~"Tetris" Block optimizations~~

    * ~~MEM OPTS: You need to first implement kernel fusion, but after that, you can remove entire allocations altogether~~
        * ~~if alloc + dealloc is in one fused op, delete alloc, and then just use a "temporary var"~~

    * <mark>Host --> Device feeder</mark>
        * fix this before implementing the x86 implementation
    
    * Divergent branching conflicts
        * Could be more complicated on various scenarious (sources, vars, concat, trackers, etc.)        
        * you have to know what is right and what is not right to be fair.

    * Fix constant as part of contigious, view, etc.
        * also the problem with multihead attention asw

    * **MEM OPTS**: Swizzling memory
        * dependending on access patterns IF there's like a single type of access pattern
            * if multiple, you might need to just rely on those different access patterns
            * just check and whether you can optimize

        * Remove movement opt if.. 
            * alr know contigious when 1. same id 2. same access expression

    * **OPTS**: Swapping accessing expressions between input and output of two adjacent kernels?
        * is it even possible? 

    * Memory experiments needed (do this movement/no movement experiment after kernel fusion)
        * **You should test whether a weird write is slower than a fast write + movement**
            * There's specialize transpose kernels as well...
            * you could try experimenting with that.
            * is there cases where fast write + movement is better?

        * **ALSO TEST**
            * which of the following procedures is best
                * dot product (uncontigious write) --> sum --> dot product (uncontigious read)
                * dot product (uncontigious write) --> sum --> movement --> dot product (contigious read)
                * dot product (contigious write) --> movement --> sum --> movement --> dot product (contigious read)

                * etc. etc. etc. 

    * **Expression simplification**
        * similar to opt remainder %.
        * Optimize at make_minus or make etc. 
        * There might be edge cases for simplify expr func. Still keep that (need experimentation)
            * v & 63 & 63 --> v & 63

    * **Kernel Fusion**
        * ~~do basic multiple binary/unary kernel fusion~~
        * dot prod impl is kinda wacky
            * access expression assumes global id...
        * anyway to do dotprod fusion?  
            * similar to flash attention
            * look into optimized [link](https://siboehm.com/articles/22/CUDA-MMM)
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
        * There's recent work on kernel fusion of **everything** somehow
            * however, you'd have to handle your own GPU synchronization. **THIS CAN BE A BENEFIT**.

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

* **HLIR Opts**
    * ~~MAKE IT FAST~~

    * ~~Rewrite entire thing to support while and if statements rather BR statements~~
        * have the graph idea in mind as well

    * <mark>Not sure if you can improve even further/less bugs if you turn it into a GRAPH rather than a list of optimizations</mark>
        * maybe some optimizations can benefit from this, not everything...
        * `to_graph` func should be created and used across IRs that benefit from it.
            * good for debugging as well
        * I think really simple HLIR opts (like view + view, constant evaluation, etc.) can be represented as a graph and optimized from there
            * opeq as well
        * Dept optimization can also be done with a graph (is it faster? no clue)
        * repeat optimization can also be done with a graph (not faster, but just easier for traversal)
        * But... you have to flatten graph and then put on prox_rev_opt 

    * Prox opt doesn't have safegaurd for IF/While? (tests still work surprisingly)

    * Set contigious operations of var deps at the end of the program
        * Some contigious IR opts can be done
        * If two contigious in series (or seperated but referencing same var)
            * remove the first contigious operation
        * Determine if it needs to be contigious in the first place.

    * Concat + view operations can be streamlined
        * this is mostly due to **concat**. If I am being honest, there's probably a better way for implementing backward pass for concat? I believe
            * maybe not 
        * Should be replaced in graphical format
        * r instruction?
        * main problem with transformer implementation right now (and well, any other implementations)

    * View removal
        * If multiple views in sequence, just turn it into the one single view (the last view operation)
        * if view is already in shape, then delete

    * Remove double permutations
        * if `b = a.permute([1, 0])` followed by `c = b.permute([1, 0])`, this is the same as the original input `a`
        * transpose via transpose

    * Constant evaluator: including 0 and 1 tracking
        * 0 * val --> 0; optimize
        * 1 * val --> val; optimize
        * 0 + val --> val; optimize
        * if a = 2, b = 3, and c = a * b, then set to 6

    * More aggressive IR optimizations for localization:
        * Also test for RNN, Transformers --> improve library
        * Then you can improve the IR optimizations as well. 
            * etc. etc. etc. 

* **General Ideas**: 
    * Dynamic Shape

    * Cuda graphs
        * have to use BR compiler hints in order to determine while or if statement
        * you may also need to send extra compiler hints 
        * skdjfksjdfkjsdkfjskdfjksdjf that's also going to be pretty weird.

## Rust Codebase

* Remove excessive clones (Ctrl+shift+F --> find)
    * instead of strings as ids, just use usize and instead try to have String only appear at display/debug
    * less clones --> usize implements copy traits
* ~~Put `IndexMap<String, Vec<IRCmds>>` under a struct (represents HLIR cmds)~~
* ~~we iter over `(block_name, b_cmds) in cmds.iter()... for cmd in b_cmds` a lot...~~
    * ~~but can this be really done? we have to trace the BR graph as we do matrix tracker, etc.~~
* Rebrand from IR to "HLIR"
* Use macros for repetitive statements
    * example, `kernel/to_instr` can be simplified to macros
    * in general the node trait is highly repeatable
    * fusion (dp elw) is also highly repetitive
* Especially for optimizations, there are some commonly shared functions 
    * like insert X items into V locations
* better debug messages (especially in frontend)

## Ignored

* **ADVANCED**: Allow dot product and sum to be accessed WITHOUT CONTIGIOUS
    * This is a problem because dot produce/reduce kernels often have `load` instructions that load 4 bytes + orig
        * sometimes, it just needs to load ONE (think)
    * this needs some kernel variant of some sort, which is really weird. So for now, we have to assume contigious.
    * technically, you only need the weight tensor to be in full contigious, as the value in the x value is just constant
        * look at the implementation :D
        * This is the only case for the CPU. NVIDIA / etc. implementations might be different.
        * per device, you probably need to provide whether the support for certain fusion implementations is common.
            * probably pass it as param to `to_kernel`
    * **Why ignored? This is rarely the case**
        * Sum and dot prod are ALMOST never after broadcasting 
        * To be specific, the weight matrix is almost never broadcasted. 
            * X is **not required** to be contigious, which can very well be broadcasted.
        * A broadcast and a sum along the same direction is **the exact same** as ELW mult
            * this optimization can be done at HLIR level, but seriously that's just a dumb thing to do...
        * broadcast and sum along the different direction can just be reversed I believe?
            * sum --> broadcast
            * only tested in 2d case, not sure if extends anywhere else.