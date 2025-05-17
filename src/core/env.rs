// flags that changes the behavior

pub mod env_flags {
    pub fn disable_ir_opt () -> bool {
        if let Ok(val) = std::env::var("IROPT") { if val == "0" { return true } }
        false 
    }
}