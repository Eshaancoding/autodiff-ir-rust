# Autodiff-IR Rust Library

Converts to a full computation forward/backward graph to IR to be executed.

Syntax is simple: 

```rust
fn nn () {
    autodiff::set_irbuilder(autodiff::TensorRsIRBuilder::new());

    let l1 = nn::Linear::new(5, 3, false);
    let l2 = nn::Linear::new(3, 2, true);
    
    let x = autodiff::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.2, 6.3], 
        vec![2, 5]
    );

    let res = nn::sigmoid(l1.f(x));
    let res = l2.f(res);
    
    res.forward();
    autodiff::ir_print();
    autodiff::execute();    
}
```

`.forward` and `.backward` just adds instructions. `autodiff::execute()` executes the instructions. Then, the `.get()` in `res.val().get()` retrieves the gradient from the IR/Device and returns the data value

## Codebase Organization

### Core
Core definitions and functionality for the autodiff library
1. `autodiff.rs` --> Stores helpful functions for the end user to create Nodes / set IR
2. `constant.rs` --> Constant declaration (ex: 3.0 in 3.0 * x, where x is Node)
3. `ir.rs` --> general definitions of IR interface.
4. `node.rs` --> general definition of node; central building block towards computation graph.
5. `tensor.rs` --> Implement input value for the computation graph (as oppose to constants). 
6. `value_data.rs` --> Class used to return data after IR has been executed
7. `value.rs` --> General Class used while propogating forward/backward to create IR.

### Graph 
Implementation of core operations used to construct the computation graph. File names are self-explanatory. As the computation graph is traversed (at `.forward()` and `.backward()`), it will add the necessary IR 

### NNs
Defines neat functions used in machine learning operations. Examples are: `Linear`, `SGD` optimizer, etc. Note that this still uses the core operations defined in `Graph`.

### IR
As explained before, IR is added as the computation graph is traversed. The `ir` folder contains handling how the IR is appended and many important optimizations are applied. After these optimizations are applied, the resultant IR will be sent to the corresponding device.

### Devices 
Defines the devices the computation can run on. Note that since most devices are inherently "thread-like" (running compiled instructions on individual threads), most devices will use constructs from the `kernel` folder before execution.

Currently supports:
1. OpenCL
    * uses `cl3` 
2. CUDA (Soon)
    * uses a wrapper around the CUDA toolkit: `cudarc`
3. Metal (Soon)
4. AVX (Soon?)

If you would like to implement your own backend, all you have is override the `Device` trait (defined in `core/ir.rs`, should be in seperate .rs file though). 
There's only two functions, `execute` and `get_tensor`
* `execute` the `cmds`, which is a map between the block name (we have some basic control constructs) and vector of `IRCmds`. 
    * note that we provide the `kernel` folder, which can convert `cmds` into a list of `Kernel` constructs, which can be helpful for later.

### Kernel
Most devices are thread-like. Kernels define what computation to run at each thread. So, this folder aims to be give general constructs over the needed kernels used to run a IR. 

After defining these constructs, we can also perform **kernel fusion** and **parameter optimization**.
    * this is not implemented yet, but definitely will be.

After the kernels are constructed + optimized, it will be sent to the respective device to be executed.

## Features

**Tensor Manipulation**
1. Concatenation
2. Indexing/Splicing
3. Data Permutations
4. View/reshape
5. Broadcasting

**Element Wise (ELW) Operations**
1. *, /, +, -
2. *=, /=, +=, -= w/ grad/val operation support 
3. `.sum(dim)`
4. `.pow(val)`. Note that val is a float
5. exponential, sin, and cos
6. Equalities:
    * ==, >, <
    * `.all()` functionality

**Linear Algebra Operations**:
1. Dot Product

**Machine Learning**:
1. Dense Layer
2. Sigmoid activation function
3. SGD

**Autograd Features**:
1. Ability to use grad/forward values to alter orig/other variables (Helpful for optimizer)
2. Multiple .backward w/ gradient accumulation
3. Reset gradient & disable grad using `.detach()`

**Control**:
Our IR also consists of basic control, including
1. Blocks (add code under these blocks)
2. If statements 
3. For loops
4. While loops

## To do
* In general:
    * use references instead of excessive clones

========== To complete the autodiff lib ========== 
==== All of this will prolly be completed at Stanford itself ====

* Softmax 
* ReLU (grad would be interesting; maybe add a piecewise forward node or something)
* Different losses: 
    * Cross entropy
    * MSE
    * etc.
* RmsNorm   <-- Needed for Transformer
* LayerNorm <-- Needed for Transformer
* Dropout   <-- Dropout
* All of the above operations should be enough for building up to attention
* Transformer
* Implement Adam
* batch matrix multiplication
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

* DataLoader

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

## IR Optimization

### LN (linear algbera) IR
* ~~Memory stop redudancy~~
    * ~~Then after that, you can turn into +=, *=, etc. (last stage of pipeline)~~

* ~~Graph pruning for useless/unread values~~
    * ~~If ids will never read and are still be computed/no dependency, then don't include the operation at all~~
    * ~~Need to turn a sequence of instructions into a graph first~~

* View removal
    * If multiple views in sequence, just turn it into the one single view (the last view operation)

* Remove double permutations
    * if `b = a.permute([1, 0])` followed by `c = b.permute([1, 0])`, this is the same as the original input `a`

* ~~Duplicate operations / branches~~
    * ~~if the operation references same application and operation, then just keep one & calculate~~
    * ~~note that this can be for entire branches (sum, sum, view) or (view, broadcast, broadcast);~~
        * ~~find some way to graphically/algorithmically find this~~

* Constant evaluator: including 0 and 1 tracking
    * 0 * val --> 0; optimize
    * ~~1 * val --> val; optimize~~
    * 0 + val --> val; optimize
    * if a = 2, b = 3, and c = a * b, then set to 6

* More aggressive IR optimizations for localization:
    * Also test for RNN, Transformers --> improve library
    * Then you can improve the IR optimizations as well. 
        * etc. etc. etc. 

* ~~Put references as close as possible to decl~~
    * ~~Proximity optimization~~
    * ~~Good for future for memory.~~
    * ~~bottom->up opt~~
    * ~~up->bottom opt~~

## Why Rust?

Put simply, I can write a lot less code with Rust rather than C++. I love the organization of files in Rust. Plus, the memory safety features are a big plus (although, I do abuse them somewhat with `Rc<RefCell>`). 

Admittedly, it took a while for me to organize the overall organization of this repository (took over 3 rewrites). But after that, it was smooth sailing from there.

However, some of Rust is a hindrance. Namely, the fact that AVX512 are not fully supported (even if it is, I would have to use nightly rust). I would love [that feature](https://github.com/rust-lang/rust/issues/111137). Second, most underlying backends - NVIDIA Toolkit, OpenCL, etc. - *anyways* use C++ (who said safety was guaranteed in this repo?). Hell, I am generating kernels written in C++ syntax within rust. This is not necessarily a "hindrance", but it's kinda funny to me.