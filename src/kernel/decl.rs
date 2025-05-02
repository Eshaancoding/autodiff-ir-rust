#[derive(Clone)]
pub enum Value {
    Constant { val: f64 },
    BlockX,  // block x - unique to each thread 
    BlockY,  // block y - unique to each thread
    ThreadX, // thread X - unique to each thread
    ThreadY, // thread Y - unique to each thread.
    Global,  // global idx - unique to each thread
}

// shorthand expressions
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
    pub id: String,        // id of the matrix (we replace this id with a pointer at device)
    pub access: Expression // note that access expressions can vary between the type of kernels.
}


// input matrix could be two types: a matrix or a matrix concat (if matrix concat, then we have to access two seperate memory locations within kernel)
#[derive(Clone)]
pub enum InputMatrix {
    IMatrix { mat: Matrix },
    // if there's a concat, then there's references to two matrix if that makes sense
    ConcatMatrix { 
        id_one: Box<InputMatrix>, 
        id_two: Box<InputMatrix>, 
        access: Expression
    }, 
}

#[derive(Clone)]
pub enum BinaryInput {
    // for binary op; could also be a constant 
    Constant { val: f64 }, 
    IMatrix { input_matrix: InputMatrix } 
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
        a: BinaryInput,
        b: BinaryInput,
        res: Matrix,
        op: BinaryOp,
    },  

    // reduce kernels, specifically along a dim (ex: sum)
    Reduce { 
        a: InputMatrix,
        res: Matrix,
        op: ReduceOp
    },  

    // dot product kernels
    DotProd { 
        a: InputMatrix,
        b: InputMatrix,
        res: Matrix
    }, 

    // unary (single function) kernels. Such as exp2, negative, recip, etc.
    Unary { 
        a: InputMatrix,
        res: Matrix,
        op: UnaryOp
    },  

    // kernels that specifies movement / permutations / concatenation of tensors
    // note that this movement kernel is not initially created; it's generated during optimizations. Otherwise, we use fancy index calculations.
    // we test whether it's optimal to use movement tensor or use the allocated memory locations weirdly. 
    Movement { 
        a: InputMatrix,
        res: Matrix
    }
}

