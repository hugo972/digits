use crate::nural::activation_layer::{ActivationLayer, ActivationLayerKind};
use crate::nural::nural_network::{NuralNetwork, NuralNetworkLossKind};
use crate::nural::transform_layer::TransformLayer;
use crate::utils::shuffle_iter::ShuffleIterExt;
use std::fs::File;
use std::io::Read;

pub const DIGIT_COUNT: usize = 1000;
pub const DIGIT_SIZE: usize = 28;
pub const DIGIT_BUFFER_SIZE: usize = DIGIT_SIZE.pow(2);

pub fn digit_network() {
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

    // learn(&digits);

    let nural_network = NuralNetwork::load_file("./data/digits.tnn").unwrap();

    for digit in 0..=9 {
        let digit_variant = rand::random_range(501..1000);
        let digit_data = digits[digit][digit_variant]
            .iter()
            .map(|&d| d as f64 / 255.0)
            .collect::<Vec<f64>>();

        let output = nural_network.predict(digit_data.as_slice());

        let predicted_digit = output
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| (**d * 100.0) as i32)
            .unwrap()
            .0;
        println!(
            "Actual: {} [{}] Prediction: {}",
            digit, digit_variant, predicted_digit
        );
    }
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
                .take(100)
                .map(|d| {
                    (
                        d.iter()
                            .map(|&d| d as f64 / 255.0)
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

pub fn get_digits(path: &str) -> Vec<Vec<u8>> {
    let mut file = File::open(path).unwrap();
    let mut image_data = vec![0u8; DIGIT_COUNT * DIGIT_BUFFER_SIZE];
    file.read(&mut image_data).unwrap();
    image_data
        .chunks(DIGIT_BUFFER_SIZE)
        .map(|d| d.to_vec())
        .collect()
}
