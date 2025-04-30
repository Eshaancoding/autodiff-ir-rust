/*
Get score of cmds by the chain size and length
We prioritize the length of the chain over the number of chains there are.
However, the less the number of chains, the better (thus the - chain_len)
*/

use indexmap::IndexMap;
use crate::IRCmds;
use super::ir_to_res;

fn number_of_digits(n: usize) -> u32 {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as u32 + 1
    }
}

pub fn get_score (cmds: &IndexMap<String, Vec<IRCmds>>) -> usize {
    let mut chain_size: usize = 0;
    let mut chain_len: usize = 0;
    let mut max_instr_len: usize = 0;

    let block_names: Vec<String> = cmds.iter().map(|(f, _)| f.clone()).collect();
        
    for block_name in block_names.iter() {
        let mut chain: Vec<(usize, usize)> = vec![];
        let mut init_ch: Option<(String, usize)> = None;
        let instr_list = cmds.get(block_name).unwrap();
        
        if instr_list.len() > max_instr_len { max_instr_len = instr_list.len(); }

        for (idx, cmd) in instr_list.iter().enumerate() {
            if let Some(result) = ir_to_res(cmd.clone()) {
                match &init_ch {
                    None => { 
                        init_ch = Some((result, idx));
                    },
                    Some((start_res, start_idx)) => {
                        if result != *start_res {
                            if *start_idx != idx-1 { chain.push((*start_idx, idx-1)); }
                            init_ch = Some((result, idx));
                        }
                    }
                } 
            }
        }

        for (start, end) in chain.iter() {
            chain_size += end - start;
        }
        chain_len += chain.len();
    }

    let ttl = 10usize.pow(number_of_digits(max_instr_len));

    (chain_size * ttl) + (ttl - chain_len)
}