/*
Implements displays for everything in the kernel folder
Extremely helpful for debugging everything (kernel fusion, expression, trackers, etc.)
*/

use std::fmt::Debug;
use std::io::{self, Write};
use std::{collections::HashMap, fmt::Display};
use super::{kernel_decl::{Expression, Input, Matrix, Value}, trackers::AllocTracker};

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
                write!(f, "{}", val)
            },
            Value::Global => {
                write!(f, "#global")
            },
            Value::X => {
                write!(f, "#x")
            },
            Value::Y => {
                write!(f, "#y")
            },
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix with id: {} and access: {}", self.id, self.access) 
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

// input interface for console
pub fn console_hashmap<T: Debug> (hmap: &HashMap<String, T>) {
    loop {
        print!("> ");
        io::stdout().flush().expect("Can't flush output");
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("Can't read input from user");

        if user_input.trim() == "q" { break; }

        if let Some(res) = hmap.get(user_input.trim()) {
            println!("{:#?}", res);
        } else {
            println!("Not found in hashmap");
        }
    }
}

pub fn console_list<T: Debug> (list: &Vec<T>) {
    loop {
        print!("> ");
        io::stdout().flush().expect("Can't flush output");
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).expect("Can't read input from user");

        if user_input.trim() == "q" { break; }

        let parsed_index = match user_input.trim().parse::<usize>() {
            Ok(v) => v,
            _ => { 
                println!("Can't parse into usize");
                continue;
            }
        };

        if let Some(res) = list.get(parsed_index) {
            println!("{:#?}", res);
        } else {
            println!("Index out of bounds");
        }
    }
}