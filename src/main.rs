#![allow(dead_code)]

use crate::nural::activation_layer::{ActivationLayer, ActivationLayerKind};
use crate::nural::nural_network::{NuralNetwork, NuralNetworkLossKind};
use crate::nural::transform_layer::TransformLayer;
use crate::utils::bin::{get_bin_digits, get_digits, print_bin_digit};
use crate::utils::shuffle_iter::ShuffleIterExt;

mod nural;
mod utils;

fn main() {
    digit_network()
}

fn digit_network() {
    let digits = [
        get_digits("./data/data0.bin"),
        get_digits("./data/data1.bin"),
        get_digits("./data/data2.bin"),
        get_digits("./data/data3.bin"),
        get_digits("./data/data4.bin"),
        get_digits("./data/data5.bin"),
        get_digits("./data/data6.bin"),
        get_digits("./data/data7.bin"),
        get_digits("./data/data8.bin"),
        get_digits("./data/data9.bin"),
    ];

    //learn(&digits);

    let nural_network = NuralNetwork::load_file("./data/digits.tnn").unwrap();
    println!("{}", nural_network);

    for digit in 0..=9 {
        let digit_variant = rand::random_range(501..1000);
        let digit_data = digits[digit][digit_variant]
            .iter()
            .map(|&d| d as f64)
            .collect::<Vec<f64>>();

        let output = nural_network.predict(digit_data.as_slice());

        let predicted_digit = output
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| (**d * 100.0) as i32)
            .unwrap()
            .0;
        println!(
            "Actual: {} [{}] Prediction: {} {:?}",
            digit, digit_variant, predicted_digit, output
        );
    }

    fn learn(digits: &[Vec<Vec<u8>>; 10]) {
        let mut nural_network = NuralNetwork::new(
            vec![
                Box::new(TransformLayer::new(28 * 28, 40)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
                Box::new(TransformLayer::new(40, 10)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
            ],
            0.1,
            NuralNetworkLossKind::Mse,
        );

        let data = digits
            .iter()
            .enumerate()
            .flat_map(|(digit, digit_data)| {
                let mut output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                output[digit] = 1.0;

                digit_data
                    .iter()
                    .take(500)
                    .map(|d| {
                        (
                            d.iter()
                                .map(|&d| d as f64)
                                .collect::<Vec<f64>>()
                                .as_slice()
                                .to_owned(),
                            output.as_slice().to_owned(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .shuffle()
            .collect::<Vec<_>>();

        nural_network.train(data.as_slice(), 100);
        nural_network.save_file("./data/digits.tnn").unwrap();
    }
}

fn bin_digit_network() {
    let digits = [
        get_bin_digits("./data/data0.bin"),
        get_bin_digits("./data/data1.bin"),
        get_bin_digits("./data/data2.bin"),
        get_bin_digits("./data/data3.bin"),
        get_bin_digits("./data/data4.bin"),
        get_bin_digits("./data/data5.bin"),
        get_bin_digits("./data/data6.bin"),
        get_bin_digits("./data/data7.bin"),
        get_bin_digits("./data/data8.bin"),
        get_bin_digits("./data/data9.bin"),
    ];

    learn(&digits);

    let nural_network = NuralNetwork::load_file("./data/bin_digits.tnn").unwrap();

    let digit = 5;
    let digit_variant = 534;
    let digit_data = digits[digit][digit_variant];

    /* let digit_data = [
        0b0000000000000000000000000000,
        0b0000001110000000011100000000,
        0b0000011110000001111000000000,
        0b0000011110000001111000000000,
        0b0000111100000011110000000000,
        0b0001111000000111110000000000,
        0b0001111000000111100000000000,
        0b0001111000000111100000000000,
        0b0001110000001111000000000000,
        0b0011110000001111000000000000,
        0b0011110000001111000000000000,
        0b0011110000001111000000000000,
        0b0011111000011110000000000000,
        0b0001111111111110000000000000,
        0b0000111111111110000000000000,
        0b0000000000011110000000000000,
        0b0000000000001110000000000000,
        0b0000000000001110000000000000,
        0b0000000000001110000000000000,
        0b0000000000001111000000000000,
        0b0000000000001111000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
    ];*/

    print_bin_digit(digit_data);
    let output = nural_network.predict(
        digit_data
            .iter()
            .map(|&d| d as f64)
            .collect::<Vec<f64>>()
            .as_slice(),
    );
    println!("Prediction: {:?}", output);

    fn learn(digits: &[Vec<[i32; 28]>; 10]) {
        let mut nural_network = NuralNetwork::new(
            vec![
                Box::new(TransformLayer::new(28, 40)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
                Box::new(TransformLayer::new(40, 10)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
            ],
            0.1,
            NuralNetworkLossKind::Mse,
        );

        let data = digits
            .iter()
            .enumerate()
            .flat_map(|(digit, digit_data)| {
                let mut output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                output[digit] = 1.0;

                digit_data
                    .iter()
                    .take(500)
                    .map(|d| {
                        (
                            d.iter()
                                .map(|&d| d as f64)
                                .collect::<Vec<f64>>()
                                .as_slice()
                                .to_owned(),
                            output.as_slice().to_owned(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .shuffle()
            .collect::<Vec<_>>();

        nural_network.train(data.as_slice(), 100);
        nural_network.save_file("./data/bin_digits.tnn").unwrap();
    }
}

pub fn xor_network() {
    // learn();

    let nural_network = NuralNetwork::load_file("./data/xor.tnn").unwrap();

    println!("trained results:");

    println!("0 xor 0 = {:?}", nural_network.predict(&[0.0, 0.0]));
    println!("0 xor 1 = {:?}", nural_network.predict(&[0.0, 1.0]));
    println!("1 xor 0 = {:?}", nural_network.predict(&[1.0, 0.0]));
    println!("1 xor 1 = {:?}", nural_network.predict(&[1.0, 1.0]));

    fn learn() {
        let mut nural_network = NuralNetwork::new(
            vec![
                Box::new(TransformLayer::new(2, 3)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
                Box::new(TransformLayer::new(3, 2)),
                Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
            ],
            0.1,
            NuralNetworkLossKind::Mse,
        );

        nural_network.train(
            &[
                (vec![0.0, 0.0], vec![1.0, 0.0]),
                (vec![0.0, 1.0], vec![0.0, 1.0]),
                (vec![1.0, 0.0], vec![0.0, 1.0]),
                (vec![1.0, 1.0], vec![1.0, 0.0]),
            ],
            1000,
        );

        nural_network.save_file("./data/xor.tnn").unwrap();
    }
}
