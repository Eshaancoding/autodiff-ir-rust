/*
Implements displays for everything in the kernel folder
Extremely helpful for debugging everything (kernel fusion, expression, trackers, etc.)
*/

use std::fmt::{Display, Formatter};
use crate::{create_debug_hlp_funcs, trackers::Location};

use super::{
    kernel_decl::{Expression, Input, Matrix, Value, KernelProcedure, Kernels}, 
    trackers::AllocTracker
};
use colored::Colorize;

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Add { a, b } => {
                write!(f, "({} + {})", a, b)
            },
            Expression::Minus { a, b } => {
                write!(f, "({} - {})", a, b)
            },
            Expression::Mult { a, b } => {
                write!(f, "({} * {})", a, b)
            },
            Expression::Div { a, b } => {
                write!(f, "({} / {})", a, b)
            },
            Expression::Remainder { a, b } => {
                write!(f, "({} % {})", a, b)
            },
            Expression::ShiftLeft { a, b } => {
                write!(f, "({} << {})", a, b)
            },
            Expression::ShiftRight { a, b } => {
                write!(f, "({} >> {})", a, b)
            },
            Expression::BitwiseAnd { a, b } => {
                write!(f, "({} & {})", a, b)
            },
            Expression::Val { v }  => {
                write!(f, "{}", v)
            },
            Expression::MoreThan { a, b } => {
                write!(f, "({} > {})", a, b)
            },        
            Expression::LessThan { a, b } => {
                write!(f, "({} < {})", a, b)
            },        
        }
    }
}

pub fn display_vec<T> (v: &Vec<T>) where T: Display {
    println!("[\n{}\n]", v.iter().map(|e| "\t".to_string() + &e.to_string()).collect::<Vec<_>>().join(",\n")) 
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Constant { val } => {
                write!(f, "{}", val.to_string().green())
            },
            Value::Global => {
                write!(f, "{}", "#global".green())
            },
            Value::X => {
                write!(f, "{}", "#x".green())
            },
            Value::Y => {
                write!(f, "{}", "#y".green())
            },
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "M (id: {}, access: {})", self.id, self.access) 
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Input::Constant { val } => {
                write!(f, "CS (V: {})", val.to_string().green())
            },
            Input::Mat { mat } => {
                write!(f, "{}", mat)
            }
            Input::ConcatMatrix { id_one, id_two, conditional } => {
                write!(f, "C (a: {}, b: {}, cond: {})", id_one, id_two, conditional)
            }
        }
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(block_id: {}, loc: {})", self.proc_id, self.loc)
    }
}

impl<'a> Display for AllocTracker<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = write!(f, "Alloc Tracker:\n");
        for (ir_name, alloc_entry) in self.vars.iter() {
            let _ = write!(f, "\n");
            let _ = write!(f, "\tIR name \"{}\":\n", ir_name);
            let _ = write!(f, "\t-> Alloc ID: {}\n", alloc_entry.id);
            let _ = write!(f, "\t-> Size: {}\n", alloc_entry.size);
            let _ = write!(
                f, 
                "\t-> Lifetime: {} --> {}\n", 
                alloc_entry.alloc_loc, 
                alloc_entry.dealloc_loc.as_ref().map_or("None".to_string(), |v| format!("{}", v).to_string())
            );
            let _ = write!(f, "\t-> Has initial content: {}\n", alloc_entry.initial_content.is_some());
            if let Some(v) = alloc_entry.initial_content.as_ref() {
                let _ = write!(f, "\t-> Initial content size: {}\n", v.len());
            }
        }

        write!(f, "")
    }
}

// ======================================= Kernel Procedure printing ======================================= 
create_debug_hlp_funcs!(Kernels, KernelProcedure);

impl Display for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernels::EX => {
                let _ = write!(f, "{}", "EXIT".red());
            },
            Kernels::Unary { a, res, op, size } => {
                let _ = write!(f, "{} {} {} ({})", res, " = ".on_blue(), format!(" {:#?} ({}) ", op, size.to_string().yellow()).bold(), a);
            },
            Kernels::Binary { a, b, res, op, size } => {
                let _ = write!(f, "{} {} {} {} {}", res, " = ".on_blue(), a, format!(" {:#?} ({}) ", op, size.to_string().yellow()).bold(), b);
            },
            Kernels::Reduce { a, res, op, vec_size, reduce_size} => {
                let _ = write!(f, "{} {} {} ({})", res, " = ".on_blue(), format!(" {:#?} (Vec/X: {}, Reduce/Y: {}) ", op, vec_size.to_string().yellow(), reduce_size.to_string().yellow()).bold(), a);
            },
            Kernels::DotProd { a, b, res, batch_size, input_size, output_size } => {
                let b_yel = batch_size.to_string().yellow();
                let i_yel = input_size.to_string().yellow();
                let o_yel = output_size.to_string().yellow(); 
                let _ = write!(f, "{} {} {} {} {}", res, " = ".on_blue(), a, format!(" ({}x{} DP {}x{})", b_yel, i_yel, i_yel, o_yel).bold(), b);
            },
            Kernels::Movement { a, res , size} => {
                let _ = write!(f, "{} {} {}", res, format!(" <-(Move {})- ", size.to_string().yellow()).bold(), a);
            },
            Kernels::While { .. } => {
                print_while(f, self, 0);
            },
            Kernels::If { .. } => {
                print_if(f, self, 0);
            },
            Kernels::Alloc { id, size, content, .. } => {
                let _ = write!(f, "{} {} {}", "Alloc".green().bold(), id, size.to_string().yellow().bold());
                if let Some(c) = content {
                    let _ = write!(f, " (with content of size: {})", c.len().to_string().yellow().bold());
                }
            },
            Kernels::Dealloc { id, size } => {
                let _ = write!(f, "{} {} {}", "Dealloc".red().bold(), id, size.to_string().yellow().bold());
            },
            Kernels::ElwExpr { kernels, size} => {
                let _ = write!(f, "ElwExpr Kernel Fusion ({})\n\n", size.to_string().yellow().bold());
                for k in kernels.iter() {
                    let _ = write!(f, "\t{}\n", k);
                }
            },
            Kernels::DPElwExpr { kernels, batch_size, input_size, output_size } => {
                let b_yel = batch_size.to_string().yellow();
                let i_yel = input_size.to_string().yellow();
                let o_yel = output_size.to_string().yellow(); 
                let elw_size_yel = (batch_size*output_size).to_string().yellow();
                let _ = write!(f, "DP + Elw Kernel Fusion {}\n\n", format!("({}x{} DP {}x{}) -(elw)-> {}", b_yel, i_yel, i_yel, o_yel, elw_size_yel).bold());
                for k in kernels.iter() {
                    let _ = write!(f, "\t{}\n", k);
                }
            },
            Kernels::ReduceElwExpr { kernels, vec_size, reduce_size } => {
                let v_yel = vec_size.to_string().yellow();
                let extra_info = format!("(Vec/X: {}, Reduce/Y: {}) -(elw)-> {}", v_yel, reduce_size.to_string().yellow(), v_yel).bold();
                let _ = write!(f, "Reduce + Elw Kernel Fusion {}\n\n", extra_info);
                for k in kernels.iter() {
                    let _ = write!(f, "\t{}\n", k);
                }
            },
        }
        write!(f, "")
    }
}

