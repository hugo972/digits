use crate::nural::activation_layer::{ActivationLayer, ActivationLayerKind};
use crate::nural::nural_network::{NuralNetwork, NuralNetworkLossKind};
use crate::nural::transform_layer::TransformLayer;

pub fn xor_network() {
    // learn();

    let nural_network = NuralNetwork::load_file("./data/xor.tnn").unwrap();

    println!("trained results:");

    println!("0 xor 0 = {:?}", nural_network.predict(&[0.0, 0.0]));
    println!("0 xor 1 = {:?}", nural_network.predict(&[0.0, 1.0]));
    println!("1 xor 0 = {:?}", nural_network.predict(&[1.0, 0.0]));
    println!("1 xor 1 = {:?}", nural_network.predict(&[1.0, 1.0]));
}

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
