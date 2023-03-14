import torch
import softmax
import mlp


if __name__ == "__main__":
    mlp.bench_inference(5000, "cuda")
