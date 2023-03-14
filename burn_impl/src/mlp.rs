use crate::bench::bench;
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Tensor, ElementConversion},
};
use burn_autodiff::ADBackendDecorator;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layers: Param<Vec<Linear<B>>>,
}

impl<B: Backend> MLP<B> {
    pub fn new(n_layers: usize, d_model: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| Linear::new(&LinearConfig::new(d_model, d_model)))
            .collect::<Vec<_>>();

        Self {
            layers: Param::from(layers),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for layer in self.layers.iter() {
            x = layer.forward(x);
            x = activation::relu(x);
        }

        x
    }
}

// CPU 100 iterations
// Time 8.043 s
// RAM: 576 M
//
// GPU 5000 iterations
// Time: 12.437 s
// Memory: 1222 MiB
pub fn bench_training<B: Backend>(iterations: usize, device: B::Device) {
    let random = Tensor::<ADBackendDecorator<B>, 2>::random(
        [128, 2048],
        burn::tensor::Distribution::Uniform(0.elem(), 1.elem()),
    )
    .to_device(&device);

    let model = MLP::<ADBackendDecorator<B>>::new(12, 2048).to_device(&device);
    let random = random.require_grad();

    bench(|| {
        for i in 0..iterations {
            let x = model.forward(random.clone());
            let _grads = x.sum().backward();
            println!("{i}");
        }
    });
}

// CPU 200 iterations
// Time 4.539 s
// RAM: 385 M
//
// GPU 5000 iterations
// Time: 4.021 s
// Memory: 1096 MiB
pub fn bench_inference<B: Backend>(iterations: usize, device: B::Device) {
    let random = Tensor::<B, 2>::random(
        [128, 2048],
        burn::tensor::Distribution::Uniform(0.elem(), 1.elem()),
    )
    .to_device(&device);

    let model = MLP::<B>::new(12, 2048).to_device(&device);
    let random = random.require_grad();

    bench(|| {
        for i in 0..iterations {
            let _x = model.forward(random.clone());
            println!("{i}");
        }
    });
}
