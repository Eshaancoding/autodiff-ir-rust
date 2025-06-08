use crate::kernel_decl::Expression;

pub fn calc_stride (shape: &Vec<usize>) -> Vec<Expression> {
    let n = shape.len();
    let mut strides: Vec<Expression> = vec![Expression::make_const(1); n];
    for i in (0..(n - 1)).rev() {
        strides[i] = Expression::make_const(strides[i+1].get_const().unwrap() * shape[i+1] as i32);
    }

    strides
}

pub fn global_to_ndim (index:Expression, shape: &Vec<usize>) -> Vec<Expression> {
    let strides = calc_stride(shape);

    let nd_index: Vec<Expression> = (0..shape.len())
        .map(|i| 
            Expression::make_remainder(
                Expression::make_div(
                    index.clone(),
                    strides[i].clone()
                ), 
                Expression::make_const(shape[i] as i32)
            )
        )
        .collect();

    nd_index
}

pub fn ndim_to_global (dim: &Vec<Expression>, shape: &Vec<usize>) -> Expression {
    let strides = calc_stride(shape);
    let mut global_expr = Expression::make_mult(
        dim[0].clone(),
        strides[0].clone()
    );

    for i in 1..shape.len() {
        global_expr = Expression::make_add(
            global_expr,
            Expression::make_mult(
                dim[i].clone(), 
                strides[i].clone() 
            )
        );
    }

    global_expr
}