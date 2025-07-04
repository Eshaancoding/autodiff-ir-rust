use crate::{helper::debug::display_vec, kernel_decl::Expression, trackers::DataCmds};

fn calc_stride (shape: &Vec<usize>) -> Vec<Expression> {
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

pub fn ndim_change_datacmds (ndim: &mut Vec<Expression>, data_cmds: &Vec<DataCmds>) {
    for cmd in data_cmds.iter().rev() { 
        match cmd { 
            DataCmds::Broadcast { dim, .. } => {
                ndim[*dim] = Expression::make_const(0);
            },
            DataCmds::Index { index, dim } => {
                ndim[*dim] = Expression::make_const(*index as i32);
            },
            DataCmds::Permute { p } => {
                let mut new_dim = vec![Expression::make_const(0); ndim.len()];
                for i in 0..ndim.len() {
                    new_dim[i] = ndim[p[i]].clone()
                }
                *ndim = new_dim;
            },
            DataCmds::View { sink_dim, source_dim } => {
                let global = ndim_to_global(ndim, sink_dim);
                *ndim = global_to_ndim(
                    global,
                    source_dim
                );
            },
        }
    }
}