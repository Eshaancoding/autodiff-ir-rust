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

`.forward` and `.backward` just adds instructions. `autodiff::execute()` executes the instructions. Then, the `.get()` in `res.val().get()` retrieves the gradient from the IR and returns the data value

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
==== Chip needs to be 100% complete before then (including FP units) ==== 

* add function (need to adjust chip)

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
        * etc. <-- do this when you know how to function with weird ahh funcs
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
    * 1 * val --> val; optimize
    * 0 + val --> val; optimize
    * if a = 2, b = 3, and c = a * b, then set to 6

* More aggressive IR optimizations for localization:
    * Also test for RNN, Transformers --> improve library
    * Then you can improve the IR optimizations as well. 
        * etc. etc. etc. 
    * If you can prove that you have repeated minimal memory access:
        * you can gen pitch this idea to investors.
        * Ykwim

* ~~Put references as close as possible to decl~~
    * ~~Proximity optimization~~
    * ~~Good for future for memory.~~
    * ~~bottom->up opt~~
    * ~~up->bottom opt~~
    * This is probably an open area of research --> search for that.