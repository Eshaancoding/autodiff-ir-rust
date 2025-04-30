import torch
from torch.nn import Sequential

import torch
import torch.nn as nn
import torch.optim as optim

# Define weights and biases
l1_w = torch.tensor([
    [0.03, 0.02, 0.01], 
    [0.04, 0.05, 0.063], 
    [0.01, 0.10, 0.07], 
    [0.02, 0.01, 0.08], 
    [0.05, 0.03, 0.06]
], requires_grad=True)

l1_b = torch.tensor([0.03, 0.01, 0.032], requires_grad=True)

l2_w = torch.tensor([
    [0.022, 0.016], 
    [0.007, 0.093], 
    [0.013, 0.09]
], requires_grad=True)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, l1_w, l1_b, l2_w):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(5, 3)
        self.l2 = nn.Linear(3, 2, bias=False)

        # Manually set the weights and biases
        with torch.no_grad():
            self.l1.weight.copy_(l1_w.T)
            self.l2.weight.copy_(l2_w.T)
            self.l1.bias.copy_(l1_b)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        return self.l2(x)

# Create the model and optimizer
model = NeuralNet(l1_w, l1_b, l2_w)
for i in model.parameters():
    print(i)
opt = optim.SGD(model.parameters(), lr=0.1)

# Random input
x = torch.tensor([
    [0.01, 0.02, 0.037, 0.04, 0.05],
    [0.063, 0.07, 0.08, 0.09, 0.013]
], requires_grad=True)

# Training loop
for _ in range(10):
    opt.zero_grad()
    y = model(x)
    y.sum().backward()  # Assume a dummy loss function
    opt.step()

def round_tensor (x, num_digits):
    mult = 10.0 ** num_digits
    return (x * mult).round() / mult

# Print results
print("Res out:\n", round_tensor(y.detach().numpy(), 4))
print("\nl1_w:\n", round_tensor(model.l1.weight.detach().numpy().T, 4))
print("\nl1_b:\n", round_tensor(model.l1.bias.detach().numpy().T, 4))
print("\nl2_w:\n", round_tensor(model.l2.weight.detach().numpy().T, 4))
