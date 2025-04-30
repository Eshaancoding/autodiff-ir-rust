import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define the network
neural_net = nn.Sequential(
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128, 64)
)

# Optimizer
opt = optim.SGD(neural_net.parameters(), lr=0.1)

# Random input tensor
x = torch.randn(2, 256)

# Placeholder for result
res = None

# Training loop
start_time = time.time()

print("Starting...")
for _ in range(100):
    y = neural_net(x)  # Forward pass
    opt.zero_grad()    # Clear gradients

    # For logging/debugging purposes
    # No explicit `.forward()` needed in PyTorch when calling the model

    loss = y.sum()  # dummy loss, just to trigger backward
    loss.backward() # Backprop
    opt.step()      # Update weights

    res = y

# Access the final result, round for display
res_rounded = torch.round(res.detach() * 10000) / 10000
print(f"Result: {res_rounded}")

elapsed = time.time() - start_time
print(f"elapsed: {elapsed:.4f} s")
