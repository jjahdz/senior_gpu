import torch
x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1)
X = torch.cat([x_t**3, x_t**2, x_t, torch.ones_like(x_t)], dim=1)
y_t = torch.tensor(y, dtype=torch.float32)
w = torch.linalg.lstsq(X, y_t).solution
print("Reference coefficients:", w)
print("Reference MSE:", ((X @ w - y_t)**2).mean().item())