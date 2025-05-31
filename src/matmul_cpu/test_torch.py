import torch
import time

print(torch.__config__.show())

# Matrix dimensions
N = 512
NUM_ITR = 300
NUM_TRIALS = 20

# Create two random matrices
A = torch.randn(N, N).to(torch.float32)
B = torch.randn(N, N).to(torch.float32)

# Warm-up
torch.matmul(A, B)
            
avg = 0
            
for _ in range(NUM_TRIALS):
    # Time the multiplication
    start = time.time()
    for _ in range(NUM_ITR):
        C = torch.matmul(A, B)
    end = time.time()

    # Compute the number of floating-point operations: 2 * N^3
    flops = 2 * N**3

    # Time in seconds
    elapsed_time = (end - start) / NUM_ITR

    # GFLOPS
    gflops = flops / (elapsed_time * 1e9)

    # print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Performance: {gflops:.2f} GFLOPS")

    avg += gflops

print(f"Avg gflops: {(avg / NUM_TRIALS):.3f} GFLOPS")
