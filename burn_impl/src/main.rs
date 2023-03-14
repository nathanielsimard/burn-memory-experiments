use burn_tch::{TchBackend, TchDevice};

fn main() {
    type Backend = TchBackend<burn::tensor::f16>;
    let device = TchDevice::Cuda(0);

    burn_impl::mlp::bench_inference::<Backend>(5000, device)
}
