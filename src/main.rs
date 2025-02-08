use crate::nural::activation_layer::{ActivationLayer, ActivationLayerKind};
use crate::nural::nural_network::{NuralNetwork, NuralNetworkLossKind};
use crate::nural::transform_layer::TransformLayer;
use crate::utils::bin::{get_digits, print_digit};

mod nural;
mod utils;

fn main() {
    test_xor_network()
}

fn test_digit_network() {
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

    let digits0 = get_digits("./data/data0.bin");
    /*    let digits1 = get_digits("./data/data1.bin");
    let digits2 = get_digits("./data/data2.bin");
    let digits3 = get_digits("./data/data3.bin");
    let digits4 = get_digits("./data/data4.bin");
    let digits5 = get_digits("./data/data5.bin");
    let digits6 = get_digits("./data/data6.bin");
    let digits7 = get_digits("./data/data7.bin");
    let digits8 = get_digits("./data/data8.bin");
    let digits9 = get_digits("./data/data9.bin");*/

    //let mut data: Vec<(&[f64], &[f64])> = Vec::new();
    let data = digits0
        .iter()
        .skip(900)
        .map(|d| {
            (
                d.iter()
                    .map(|&d| d as f64)
                    .collect::<Vec<f64>>()
                    .as_slice()
                    .to_owned(),
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    .as_slice()
                    .to_owned(),
            )
        })
        .collect::<Vec<_>>();

    nural_network.train(data.as_slice(), 100);

    let output = nural_network.predict(
        digits0[34]
            .iter()
            .map(|&d| d as f64)
            .collect::<Vec<f64>>()
            .as_slice(),
    );

    print_digit(digits0[34]);
    println!("Prediction: {:?}", output);
}

pub fn test_xor_network() {
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

    println!("trained results:");
    println!("0 xor 0 = {:?}", nural_network.predict(&[0.0, 0.0]));
    println!("0 xor 1 = {:?}", nural_network.predict(&[0.0, 1.0]));
    println!("1 xor 0 = {:?}", nural_network.predict(&[1.0, 0.0]));
    println!("1 xor 1 = {:?}", nural_network.predict(&[1.0, 1.0]));

    let serialized = serde_json::to_string_pretty(&nural_network).unwrap();
    let deserialized_nural_network: NuralNetwork = serde_json::from_str(&serialized).unwrap();

    println!("deserialized trained results:");
    println!("0 xor 0 = {:?}", deserialized_nural_network.predict(&[0.0, 0.0]));
    println!("0 xor 1 = {:?}", deserialized_nural_network.predict(&[0.0, 1.0]));
    println!("1 xor 0 = {:?}", deserialized_nural_network.predict(&[1.0, 0.0]));
    println!("1 xor 1 = {:?}", deserialized_nural_network.predict(&[1.0, 1.0]));
}
