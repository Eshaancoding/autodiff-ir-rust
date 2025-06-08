use crate::{
    helper::shape::{global_to_ndim, ndim_to_global}, 
    kernel_decl::Expression
};
use crate::trackers::{
    DataCmds, 
    MatrixTracker,
};

impl<'a> MatrixTracker<'a> {
    pub fn ndim_change_datacmds (&self, ndim: &mut Vec<Expression>, data_cmds: &Vec<DataCmds>) {
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
}