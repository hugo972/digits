use std::fmt;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    #[allow(dead_code)]
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn from(vec: &[f64]) -> Matrix {
        Matrix {
            cols: vec.len(),
            data: vec.to_vec(),
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        self.apply_with(other, |val, other_val| val + other_val)
    }

    pub fn apply(&self, func: impl Fn(f64) -> f64) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|val| func(*val))
            .collect();
        Matrix {
            cols: self.cols,
            data,
        }
    }

    pub fn apply_with(&self, other: &Matrix, func: impl Fn(f64, f64) -> f64) -> Matrix {
        if self.cols != other.cols && self.data.len() != other.data.len() {
            panic!("Matrix size incorrect.");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(val, other_val)| func(*val, *other_val))
            .collect();
        Matrix {
            cols: self.cols,
            data,
        }
    }

    pub fn dot_mul(&self, other: &Matrix) -> Matrix {
        self.apply_with(other, |val, other_val| val * other_val)
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows() {
            panic!("Cols must match other rows.");
        }

        let mut data = vec![0.0; self.rows() * other.cols];
        let other_t = other.transpose();
        for (row, row_vec) in self.data.chunks(self.cols).enumerate() {
            for (col, col_vec) in other_t.data.chunks(other_t.cols).enumerate() {
                data[row * other.cols + col] = dot_vec(row_vec, col_vec);
            }
        }

        Matrix {
            cols: other.cols,
            data,
        }
    }

    pub fn rnd(rows: usize, cols: usize) -> Matrix {
        let data = rand::random_iter()
            .take(rows * cols)
            .map(|val: f64| val * 2.0 - 1.0)
            .collect();
        Matrix { cols, data }
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }

    pub fn scale(&self, ratio: f64) -> Matrix {
        self.apply(|val| val * ratio)
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        self.apply_with(other, |val, other_val| val - other_val)
    }

    pub fn transpose(&self) -> Matrix {
        let mut data: Vec<f64> = vec![0.0; self.data.len()];

        let rows = self.rows();
        for row in 0..rows {
            for col in 0..self.cols {
                data[col * rows + row] = self.data[row * self.cols + col];
            }
        }

        Matrix { cols: rows, data }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut display = String::new();
        display.push('\n');

        for row_vec in self.data.chunks(self.cols) {
            display.push_str(format!("{:?}\n", row_vec).as_str());
        }

        write!(f, "{}", display)
    }
}

fn dot_vec(vec1: &[f64], vec2: &[f64]) -> f64 {
    if vec1.len() != vec2.len() {
        panic!("Vector length must be equal.");
    }

    let mut out_scl: f64 = 0.0;
    for (_, (vec1_el, vec2_el)) in vec1.iter().zip(vec2.iter()).enumerate() {
        out_scl += vec1_el * vec2_el;
    }

    out_scl
}
