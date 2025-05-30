#[derive(Clone, Debug)]
pub enum Value {
    Constant { val: i32 },
    Global,  // global idx - unique to each thread
}

// shorthand expressions
// used for index accessing, unary, and binary expression
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

    Exp2 {a: Box<Expression>},
    Log2 {a: Box<Expression>},
    Sin  {a: Box<Expression>},
    Recip {a: Box<Expression>},
    Sqrt {a: Box<Expression>},
    EqualZero {a: Box<Expression>},
    MoreZero {a: Box<Expression>},
    LessZero {a: Box<Expression>}
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
        access: Expression
    }, 
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
pub enum ComputeInstr {
    // General over binary/unary functions. Result is an "Expression".
    // temp structs over compute instructions
    Unary {
        a: Input,
        res: Matrix,
        op: UnaryOp
    },

    // binary kernels, such as a + b or a * b
    /*Expr {
        result: Expression
    },*/

    // binary kernels, such as a + b or a * b
    Binary {  
        a: Input,
        b: Input,
        res: Matrix,
        op: BinaryOp,
    },  

    // reduce kernels, specifically along a dim (ex: sum)
    Reduce { 
        a: Input,
        res: Matrix,
        op: ReduceOp
    },  

    // dot product kernels
    // note that the access expression for a, b, and res are in terms of global idx. You'd have to convert to local group --> global at dot prod kernel
    // This will be improved in the future; dot product kernels are an entire can of worms that you really don't want to enter.
    DotProd { 
        a: Input,
        b: Input,
        res: Matrix
    }, 

    // kernels that turns movement / permutations / concatenation of tensors into single contigious tensor.  
    // note that this movement kernel is not initially created; it's generated lazily or during optimizations. 
    // Note that for dot prod and reduce kernels, we need contigious as it uses single load/store operation that accesses memory address x ... x + 4.
    Movement { 
        a: Input,
        res: Matrix
    },

    // Alloc {
    //     id: String,
    //     dim: Vec<usize>        
    // },
    
    // Dealloc {
    //     id: String,
    //     dim: Vec<usize>
    // },

    // Control functions
    BR {block_id: String},
    BRE {block_id: String, a: String},
    BRZ {block_id: String, a: String},
    EX
}


