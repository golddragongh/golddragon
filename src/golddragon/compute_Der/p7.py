import torch

r = torch.tensor(30000.1, dtype=torch.float64, requires_grad=True, device="cuda")
rs = torch.tensor(.00887, dtype=torch.float64)
tau = 2 * r * torch.sqrt(r/rs) * torch.sqrt(1-rs/r)
tau.backward() # Applies the chain rule and stores in r.grad

print(r.grad) # Prints the gradient of tau with respect to r
print(r.device)