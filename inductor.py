import torch


# Define a simple function
def simple_fn(x, y):
    z = torch.matmul(x, y)
    return torch.nn.functional.softmax(z, dim=1)


# Create input tensors
x = torch.rand((100, 100))
y = torch.rand((100, 100))

# Compile the function using torch.compile() with TorchInductor backend
compiled_fn = torch.compile(simple_fn, backend="inductor")

# Call the compiled function
result = compiled_fn(x, y)

# Show the result
print("Result of compiled function:", result)
