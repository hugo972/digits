#![allow(dead_code)]

use crate::bin_digits_network::bin_digit_network;

pub mod digits_network;
mod bin_digits_network;
mod nural;
mod utils;
mod xor_network;

fn main() {
    bin_digit_network()
}