import numpy as np
import torch

N_ITERS = 50
torch.set_float32_matmul_precision("high")


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# Define a simple function
def simple_fn(x, y):
    z = torch.matmul(x, y)
    return torch.nn.functional.softmax(z, dim=1)


def generate_data():
    return (
        torch.randn((100, 100)).to(torch.float32).cuda(),
        torch.randn(100, 100).to(torch.float32).cuda(),
    )


# Compile the function using torch.compile() with TorchInductor backend
compiled_fn = torch.compile(simple_fn, backend="inductor", mode="max-autotune")

eager_times = []
for i in range(N_ITERS):
    inp = generate_data()
    with torch.no_grad():
        _, eager_time = timed(lambda: simple_fn(inp[0], inp[1]))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    inp = generate_data()
    with torch.no_grad():
        _, compile_time = timed(lambda: compiled_fn(inp[0], inp[1]))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)


eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(
    f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)
