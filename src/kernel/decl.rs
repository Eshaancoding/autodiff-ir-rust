#[derive(Clone)]
pub enum Value {
    Constant { val: i32 },
    BlockX,  // block x - unique to each thread 
    BlockY,  // block y - unique to each thread
    ThreadX, // thread X - unique to each thread
    ThreadY, // thread Y - unique to each thread.
    Global,  // global idx - unique to each thread
}

// shorthand expressions
// mostly used for index accessing
#[derive(Clone)]
pub enum Expression {
    Val {v: Value},
    Add {a: Box<Expression>, b: Box<Expression>},
    Minus {a: Box<Expression>, b: Box<Expression>},
    Mult {a: Box<Expression>, b: Box<Expression>},
    Div {a: Box<Expression>, b: Box<Expression>},
    IntDiv {a: Box<Expression>, b: Box<Expression>},
    Remainder {a: Box<Expression>, b: Box<Expression>},
    ShiftRight {a: Box<Expression>, b: Box<Expression>},
    ShiftLeft {a: Box<Expression>, b: Box<Expression>}
}

#[derive(Clone)]
pub struct Matrix {
    pub id: String,        // id of the alloc (we replace this id with a pointer at device)
    pub access: Expression // note that access expressions can vary between the type of kernels.
}

// input matrix could be two types: a matrix or a matrix concat (if matrix concat, then we have to access two seperate memory locations within kernel)
#[derive(Clone)]
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

#[derive(Clone)]
pub enum BinaryOp {
    Add,
    Multiply
}

#[derive(Clone)]
pub enum ReduceOp {
    Sum,
    Max 
}

#[derive(Clone)]
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

#[derive(Clone)]
pub enum Kernels {
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
    DotProd { 
        a: Input,
        b: Input,
        res: Matrix
    }, 

    // unary (single function) kernels. Such as exp2, negative, recip, etc.
    Unary { 
        a: Input,
        res: Matrix,
        op: UnaryOp
    },  

    // kernels that specifies movement / permutations / concatenation of tensors
    // note that this movement kernel is not initially created; it's generated during optimizations. Otherwise, we use fancy index calculations.
    // we test whether it's optimal to use movement tensor or use the allocated memory locations weirdly. 
    Movement { 
        a: Input,
        res: Matrix
    }
}

