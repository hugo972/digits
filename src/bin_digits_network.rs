use crate::digits_network::get_digits;
use crate::nural::activation_layer::{ActivationLayer, ActivationLayerKind};
use crate::nural::nural_network::{NuralNetwork, NuralNetworkLossKind};
use crate::nural::transform_layer::TransformLayer;
use crate::utils::shuffle_iter::ShuffleIterExt;

pub fn bin_digit_network() {
    let bin_digits = [
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

    learn(&bin_digits);
    //test_random(&bin_digits);
    test_concrete();
}

fn learn(bin_digits: &[Vec<[i32; 28]>; 10]) {
    let mut nural_network = NuralNetwork::new(
        vec![
            Box::new(TransformLayer::new(28, 28 * 28)),
            Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
            Box::new(TransformLayer::new(28 * 28, 40)),
            Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
            Box::new(TransformLayer::new(40, 10)),
            Box::new(ActivationLayer::new(ActivationLayerKind::Tanh)),
        ],
        0.1,
        NuralNetworkLossKind::Mse,
    );

    let bin_digit_train_data = bin_digits
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

    nural_network.train(bin_digit_train_data.as_slice(), 100);
    nural_network.save_file("./data/bin_digits.tnn").unwrap();
}

fn test_concrete() {
    let nural_network = NuralNetwork::load_file("./data/bin_digits.tnn").unwrap();
    let bin_digit = [
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000000000000000000000000,
        0b0000000011111111111100000000,
        0b0000011111111111111000000000,
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
    ];

    let bin_digit_data = bin_digit.iter().map(|&d| d as f64).collect::<Vec<f64>>();
    let output = nural_network.predict(bin_digit_data.as_slice());

    let predicted_digit = output
        .iter()
        .enumerate()
        .max_by_key(|(_, d)| (**d * 100.0) as i32)
        .unwrap()
        .0;

    print_bin_digit(bin_digit.as_slice());
    println!("Prediction: {}", predicted_digit);
}

fn test_random(bin_digits: &[Vec<[i32; 28]>; 10]) {
    let nural_network = NuralNetwork::load_file("./data/bin_digits.tnn").unwrap();

    for digit in 0..=9 {
        let digit_variant = rand::random_range(501..1000);
        let bin_digit_data = bin_digits[digit][digit_variant]
            .iter()
            .map(|&d| d as f64)
            .collect::<Vec<f64>>();

        let output = nural_network.predict(bin_digit_data.as_slice());

        let predicted_digit = output
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| (**d * 100.0) as i32)
            .unwrap()
            .0;

        print_bin_digit(bin_digits[digit][digit_variant].as_slice());
        println!(
            "Actual: {} [{}] Prediction: {}",
            digit, digit_variant, predicted_digit
        );
    }
}

pub fn get_bin_digits(path: &str) -> Vec<[i32; crate::digits_network::DIGIT_SIZE]> {
    let image_data = get_digits(path);
    let mut digits: Vec<[i32; crate::digits_network::DIGIT_SIZE]> = vec![[0; crate::digits_network::DIGIT_SIZE]; crate::digits_network::DIGIT_COUNT];
    for digit in 0..crate::digits_network::DIGIT_COUNT {
        for row in 0..crate::digits_network::DIGIT_SIZE {
            let mut d_row = 0;
            for col in 0..crate::digits_network::DIGIT_SIZE {
                if image_data[digit][row * crate::digits_network::DIGIT_SIZE + col] > 0 {
                    d_row |= 0x1;
                }

                d_row <<= 1;
            }

            digits[digit][row] = d_row;
        }
    }

    digits
}

pub fn print_bin_digit(digit: &[i32]) {
    for digit_row in digit.iter() {
        println!("{:#030b}", digit_row)
    }
}