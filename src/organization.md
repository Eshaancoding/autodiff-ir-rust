# Organization

## Tests
Multiple tests to test various parts of the automatic differentation

1. `ct.rs` --> Constant, ELW (element-wise), and function testing (cos, sqrt, etc.)
2. `evryt.rs` --> Everything else; dot product, sum, repeat, concat, unsqueeze, select/index, permute test

## Core
Core definitions and functionality for the autodiff library
1. `autodiff.rs` --> Stores helpful functions for the end user to create Nodes / set IR
2. `constant.rs` --> Constant declaration (ex: 3.0 in 3.0 * x, where x is Node)
3. `ir.rs` --> general definitions of IR interface. Implementations are in `IR` folder.
4. `node.rs` --> general definition of node; central building block towards computation graph.
5. `tensor.rs` --> Implement input value for the computation graph (as oppose to constants). 
6. `value_data.rs` --> Class used to return data after IR has been executed
7. `value.rs` --> General Class used while propogating forward/backward to create IR.

## Graph 
Implementation of different operations used to manipulate data. File names are self-explanatory.

## IR
Implementation of different IR backends that implements the backend processing
* `tensor_rs`: uses the tensor_rs library to do the basic linear algebra equations.
    * The files inside this folder are self-explanatory. `tensor_rs` is meant to be used as a toy example. Nothing *real*.
* `cuda`: Cuda backend for the project using `cudarc` <-- light wrapper over the cuda toolit
    * operator fusion?
* `opt`. This is for the IR optimizations that's invoked at `execute` (alternatively, called at `optimize`).
    * Looks at redundancies, and fixes them.

## NN
More ML/AI implementations that builds on top of the `Graph` operations.

