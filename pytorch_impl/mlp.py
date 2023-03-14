import torch
from bench import bench
from torch import nn
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_layers)
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        return x

# GPU 5000 iterations
# Time: 16.354 s
# Mem: 1204 MiB
#
# CPU 100 iterations
# Time: 7.685 s
# Mem: 708 M
def bench_training(iterations, device):
    random = torch.rand((128, 2048)).to(device)
    model = MLP(12, 2048).to(device)

    if device == "cuda":
        random = random.half()
        model = model.half()

    random.requires_grad_()

    def loop():
        for i in range(0, iterations):
            x = model(random)
            x.sum().backward()
            print(i)

    bench(loop)



# GPU 5000 iterations
# Time: 4.2037 s
# Mem: 1190 MiB
#
# CPU 200 iterations
# Time: 4.553 s
# Mem: 433 M
def bench_inference(iterations, device):
    random = torch.rand((128, 2048)).to(device)
    model = MLP(12, 2048).to(device)

    if device == "cuda":
        random = random.half()
        model = model.half()

    def loop():
        with torch.inference_mode():
            for i in range(0, iterations):
                x = model(random)
                print(i)

    bench(loop)

