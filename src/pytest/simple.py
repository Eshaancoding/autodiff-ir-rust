import torch

def simple_test():
    # Create tensors with requires_grad=True to track gradients
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]], requires_grad=True)

    y = torch.tensor([[7.0, 2.0, 6.0, 4.0],
                      [9.0, 2.0, 6.0, 4.0]], requires_grad=True)

    # Compute intermediate and final results
    t = torch.cos(2.0 * x + 3.0)
    t2 = torch.sqrt((t * y) + 3.0 * (t + x))
    res = t2 / t

    # Forward pass is done above
    # Backward pass (again, res is not a scalar, so we need to reduce)
    res.sum().backward()

    # Print results
    torch.set_printoptions(precision=4, sci_mode=False)
    print("======= Result Value =======")
    print(res.round(decimals=4))
    
    print("\n======= Gradient w.r.t x =======")
    print(x.grad.round(decimals=4))

    print("\n======= Gradient w.r.t y =======")
    print(y.grad.round(decimals=4))

simple_test()
