use crate::bench::bench;
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use burn_autodiff::ADBackendDecorator;

// CPU 100 iterations
// Time 3.425s
// RAM: 353M
//
// GPU 10000 iterations
// Time: 14.796 s
// Memory: 756 MiB
pub fn bench_inference<B: Backend>(iterations: usize, device: B::Device) {
    let random = Tensor::<B, 3>::random(
        [128, 128, 2048],
        burn::tensor::Distribution::Uniform(0.elem(), 1.elem()),
    )
    .to_device(&device);

    bench(|| {
        for i in 0..iterations {
            let _x = softmax(random.clone(), 2);
            println!("{i}");
        }
    });
}

// CPU 100 iterations
// Time 14.693
// RAM: 1047M
//
// GPU 1000 iterations
// Time: 5.365s
// Memory: 1076 MiB
pub fn bench_training<B: Backend>(iterations: usize, device: B::Device) {
    let random = Tensor::<ADBackendDecorator<B>, 3>::random(
        [128, 128, 2048],
        burn::tensor::Distribution::Uniform(0.elem(), 1.elem()),
    )
    .to_device(&device);

    let random = random.require_grad();

    bench(|| {
        for i in 0..iterations {
            let x = softmax(random.clone(), 2);
            let _grads = x.sum().backward();
            println!("{i}");
        }
    });
}

fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(tensor, dim).exp()
}

fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    tensor.clone() - tensor.exp().sum_dim(dim).log()
}
