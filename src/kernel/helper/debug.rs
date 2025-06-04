/*
Implements displays for everything in the kernel folder
Extremely helpful for debugging everything (kernel fusion, expression, trackers, etc.)
*/

use std::fmt::Display;
use super::{
    kernel_decl::{Expression, Input, Matrix, Value, Procedure, ComputeInstr}, 
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
                write!(f, "Constant with val: {}", val)
            },
            Input::Mat { mat } => {
                write!(f, "{}", mat)
            },
            Input::ConcatMatrix { id_one, id_two, access } => {
                write!(f, "Concat matrix\n\ta: {}\n\tb: {}\n with access expression: {}", id_one, id_two, access)
            }
        }
    }
}

impl Display for ComputeInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeInstr::BR { block_id } => {
                let _ = write!(f, "BR {}\n", block_id.blue());
            },
            ComputeInstr::BRE { block_id, a }  => {
                let _ = write!(f, "if {} == 1 --> BR {}\n", a, block_id.blue());
            },
            ComputeInstr::BRZ { block_id, a } => {
                let _ = write!(f, "if {} == 0 --> BRZ {}\n", a, block_id.blue());
            },
            ComputeInstr::EX => {
                let _ = write!(f, "{}", "EXIT".red());
            },
            ComputeInstr::Unary { a, res, op, size } => {
                let _ = write!(f, "{} {} {} ({})", res, " = ".on_blue(), format!(" {:#?} ({}) ", op, size.to_string().yellow()).bold(), a);
            },
            ComputeInstr::Binary { a, b, res, op, size } => {
                let _ = write!(f, "{} {} {} {} {}", res, " = ".on_blue(), a, format!(" {:#?} ({}) ", op, size.to_string().yellow()).bold(), b);
            },
            ComputeInstr::Reduce { a, res, op, size} => {
                let _ = write!(f, "{} {} {} ({})", res, " = ".on_blue(), format!(" {:#?} ({}) ", op, size.to_string().yellow()).bold(), a);
            },
            ComputeInstr::DotProd { a, b, res, batch_size, input_size, output_size } => {
                let b_yel = batch_size.to_string().yellow();
                let i_yel = input_size.to_string().yellow();
                let o_yel = output_size.to_string().yellow(); 
                let _ = write!(f, "{} {} {} {} {}", res, " = ".on_blue(), a, format!(" ({}x{} DP {}x{})", b_yel, i_yel, i_yel, o_yel).bold(), b);
            },
            ComputeInstr::Movement { a, res , size} => {
                let _ = write!(f, "{} <-(move {})- {}", res, size.to_string().yellow(), a);
            }
            // _ => todo!()
        }
        write!(f, "\n")
    }
}

impl Display for Procedure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (block_name, cmds) in self.cmd_computeinstr.iter() {
            let _ = write!(f, "\n{}:\n", block_name.blue());
            
            for (i, cmd) in cmds.iter().enumerate() {
                let _ = write!(f, "\t({}): {}", i+1, cmd);
            }
        }

        write!(f, "")        
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
            let _ = write!(f, "\t-> Has initial content: {}\n", alloc_entry.initial_content.is_some());
        }

        write!(f, "")
    }
}