import torch
from bench import bench

# CPU 100 iterations
# Time 4.739s
# RAM: 586M
#
# GPU 10000 iterations
# Time 14.738 s
# RAM: 852 MiB
def bench_inference(iterations, device):
    random = torch.rand((128, 128, 2048)).to(device)

    if device == "cuda":
        random = random.half()

    def loop():
        with torch.inference_mode():
            for i in range(0, iterations):
                _x = softmax(random, 2)
                print(i)

    bench(loop)

# CPU 100 iterations
# Time 11.180s
# RAM: 837 M
#
# GPU 1000 iterations
# Time 4.103 s
# RAM: 980 MiB
def bench_training(iterations, device):
    random = torch.rand((128, 128, 2048)).to(device)

    if device == "cuda":
        random = random.half()

    random.requires_grad_()

    def loop():
        for i in range(0, iterations):
            x = softmax(random, 2)
            x.sum().backward()
            print(i)

    bench(loop)



def softmax(tensor: torch.Tensor, dim: int):
    return log_softmax(tensor, dim).exp()

def log_softmax(tensor, dim):
    return tensor - tensor.exp().sum(dim=dim, keepdim=True).log()


