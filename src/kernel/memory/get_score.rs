/*
Get score of cmds by the chain size and length
We prioritize the length of the chain over the number of chains there are.
However, the less the number of chains, the better (thus the - chain_len)
*/

use crate::kernel_decl::KernelProcedure;

fn number_of_digits(n: usize) -> u32 {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as u32 + 1
    }
}

pub fn get_score (proc: &mut KernelProcedure) -> usize {
    let mut chain_size: usize = 0;
    let mut chain_len: usize = 0;
    let mut max_instr_len: usize = 0;

    let mut func = |proc: &mut KernelProcedure| {
        let mut chain: Vec<(usize, usize)> = vec![];
        let mut init_ch: Option<(String, usize)> = None;
        
        if proc.len() > max_instr_len { max_instr_len = proc.len(); }

        for (idx, cmd) in proc.iter().enumerate() {
            if let Some(result) = cmd.get_res() {
                match &init_ch {
                    None => { 
                        init_ch = Some((result.clone(), idx));
                    },
                    Some((start_res, start_idx)) => {
                        if *result != *start_res {
                            if *start_idx != idx-1 { chain.push((*start_idx, idx-1)); }
                            init_ch = Some((result.clone(), idx));
                        }
                    }
                } 
            }
        }

        for (start, end) in chain.iter() {
            chain_size += end - start;
        }
        chain_len += chain.len();
    };

    proc.apply(&mut func);

    let ttl = 10usize.pow(number_of_digits(max_instr_len));

    (chain_size * ttl) + (ttl - chain_len)
}