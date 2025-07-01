use std::fmt::Display;

// Value stored with Data; used primarily by user to get data from IR
#[derive(Clone, Debug)]
pub struct ValueData {
    pub dim: Vec<usize>,
    pub data: Vec<f32>,
    pub id: String,
    pub is_none: bool
}

impl ValueData {
    fn idx (&self, idx: Vec<usize>) -> f32 {
        // get global dimension given idxs; not sure about this; ALSO TEST THIS
        let mut g_idx: usize = 0;
        let mut stride: usize = 1;
        for i in (0..self.dim.len()).rev() {
            assert!(idx[i] < self.dim[i], "Index out of bounds");
            g_idx += idx[i] * stride;
            stride *= self.dim[i];
        }

        self.data[g_idx] 
    }

    pub fn none () -> ValueData {
        ValueData {
            dim: vec![],
            data: vec![],
            id: "".to_string(),
            is_none: true
        }
    }

    pub fn round (&self, num_digits: u8) -> Self {
        ValueData {
            dim: self.dim.clone(),
            data: self.data.iter().map(|x| {
                let mult = 10.0_f32.powi(num_digits as i32);
                (x * mult).round() / mult
            }).collect(),
            id: self.id.clone(),
            is_none: self.is_none
        }
    }
}

impl Display for ValueData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none {
            return write!(f, "None")
        } 

        if self.dim.len() > 2 { // lazy write if we are dealing with more than 3 dim arrays
            return write!(f, "{:#?}", self)
        }        

        let mut mt_str = "".to_string();
        mt_str += "\n[";
        if self.dim.len() == 2 {
            for x in 0..self.dim[0] {
                mt_str += "\n\t[ ";
                for i in 0..self.dim[1] {
                    mt_str += format!("{}, ", self.idx(vec![x, i])).as_str();
                } 
                mt_str.truncate(mt_str.len().saturating_sub(2));
                mt_str += "]";
            }
            
            mt_str += "\n";
        }

        if self.dim.len() == 1 {
            for i in 0..self.dim[0] {
                mt_str += format!("{}, ", self.data[i]).as_str();
            } 
            mt_str.truncate(mt_str.len().saturating_sub(2));
        }
        mt_str += "]";
        
        // finally write
        write!(f, "{}\nwith shape: {:?}", mt_str, self.dim)
    }
}