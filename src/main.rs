#![allow(dead_code)]

extern crate core;
extern crate openblas_src;

use crate::digits_network::digit_network;

mod bin_digits_network;
pub mod digits_network;
mod nural;
mod utils;
mod xor_network;

fn main() {
    digit_network()
}