use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Value {
    Constant { val: i32 },
    Global,  // global idx - unique to each thread
    X,       // X - requires the x coordinate for accessing matrix (especially in dot product)
    Y,       // Y - requires the y coordinate for accessing matrix (especially in dot product)
}

// shorthand expressions
// used for index accessing
#[derive(Clone, Debug)]
pub enum Expression {
    Val {v: Value},
    Add {a: Box<Expression>, b: Box<Expression>},
    Minus {a: Box<Expression>, b: Box<Expression>},
    Mult {a: Box<Expression>, b: Box<Expression>},
    Div {a: Box<Expression>, b: Box<Expression>},
    Remainder {a: Box<Expression>, b: Box<Expression>},
    ShiftRight {a: Box<Expression>, b: Box<Expression>},
    ShiftLeft {a: Box<Expression>, b: Box<Expression>},
    BitwiseAnd {a: Box<Expression>, b: Box<Expression>},
    MoreThan {a: Box<Expression>, b: Box<Expression>},
    LessThan {a: Box<Expression>, b: Box<Expression>}
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub id: String,        // id of the alloc (we replace this id with a pointer at device)
    pub access: Expression // note that access expressions can vary between the type of kernels. (sum vs. elw)
    // we use row-major (C++ like) and zero-indexing (C++ like)
}

// input matrix could be two types: a matrix or a matrix concat (if matrix concat, then we have to access two seperate memory locations within kernel)
#[derive(Clone, Debug)]
pub enum Input {
    Constant { val: f64 }, 
    Mat { mat: Matrix },
    // if there's a concat, then there's references to two matrix if that makes sense
    ConcatMatrix { 
        id_one: Box<Input>, 
        id_two: Box<Input>, 
        conditional: Expression // conditional on whether to access expression id_one or id_two
    }, 
}

#[derive(Clone, Debug)]
pub enum UnaryOp {
    Exp2,
    Log2,
    Sin,
    Neg,
    Recip,
    Sqrt,
    EqualZero,
    MoreZero,
    LessZero
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Multiply
}

#[derive(Clone, Debug)]
pub enum ReduceOp {
    Sum,
    Max 
}

#[derive(Clone, Debug)]
pub enum Kernels {
    // unary expressions, such as b = a.exp()
    Unary {
        a: Input,
        res: Matrix,
        op: UnaryOp,
        size: usize   // size of the input
    },

    // binary kernels, such as a + b or a * b
    Binary {  
        a: Input,
        b: Input,
        res: Matrix,
        op: BinaryOp,
        size: usize  // size of both a and b (both are same size)
    },  

    // reduce kernels, specifically along a dim (ex: sum)
    Reduce { 
        a: Input,
        res: Matrix,
        op: ReduceOp,
        vec_size: usize,    // num of vecs to be reduced (represents x)
        reduce_size: usize, // how much you have to reduce over (this represents y)
    },  

    // dot product kernels
    // note that the access expression for a, b, and res are in terms of global idx. You'd have to convert to local group --> global at dot prod kernel
    // This will be improved in the future; dot product kernels are an entire can of worms that you really don't want to enter.
    DotProd { 
        a: Input,
        b: Input,
        res: Matrix,
        batch_size: usize, // batch size (B in BxI dotprod I*O --> B*O)
        input_size: usize, // input size (I in BxI dotprod I*O --> B*O)
        output_size: usize, // output size (O in BxI dotprod I*O --> B*O)
    }, 

    // kernels that turns movement / permutations / concatenation of tensors into single contigious tensor.  
    // note that this movement kernel is not initially created; it's generated lazily or during optimizations. 
    // Note that for dot prod, we need contigious as it uses single load/store operation that accesses memory address x ... x + 4.
    Movement { 
        a: Input,
        res: Matrix,
        size: usize,   // size of the result kernel
    },

    // Alloc {
    //     id: String,
    //     dim: Vec<usize>        
    // },
    
    // Dealloc {
    //     id: String,
    //     dim: Vec<usize>
    // },

    // Control functions;
    BR {block_id: String},
    BRE {block_id: String, a: String},
    BRZ {block_id: String, a: String},
    EX
}

// list the procedure of kernels to declare
// A light wrapper for hashmap<string, Vec<ComputeInstr>>
// Implementations defined in procedure.rs
// String in HashMap<String, ...> represents the block name
// Vec<ComputeInstr> is the order of kernels to be executed
pub struct KernelProcedure {
    pub cmd_computeinstr: HashMap<String, Vec<Kernels>>
}