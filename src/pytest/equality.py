import torch

a = torch.tensor(
    [0.0, 1.0, 3.0, 2.0, 1.0, 5.0],
    requires_grad=True
)

b = 3 * (a * (a == 3.0)) ** 2
b.sum().backward();

print(b)
print(a.grad)


