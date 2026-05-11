import torch
import numpy as np

# Generate same synthetic data
n = 100000
x = torch.linspace(-1, 1, n)
y = 3.0*x**3 - 2.0*x**2 + 1.5*x + 2.0 + torch.randn(n)*0.1

# Fit with AdamW
params_adamw = torch.zeros(4, requires_grad=True)
opt = torch.optim.AdamW([params_adamw], lr=0.01, weight_decay=0.0)
for epoch in range(2000):
    pred = params_adamw[0]*x**3 + params_adamw[1]*x**2 + params_adamw[2]*x + params_adamw[3]
    loss = ((pred - y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print("AdamW:", params_adamw.detach().numpy())

# Fit with SGD
params_sgd = torch.zeros(4, requires_grad=True)
opt = torch.optim.SGD([params_sgd], lr=0.01)
for epoch in range(2000):
    pred = params_sgd[0]*x**3 + params_sgd[1]*x**2 + params_sgd[2]*x + params_sgd[3]
    loss = ((pred - y)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print("SGD:", params_sgd.detach().numpy())
print("True: [3.0, -2.0, 1.5, 2.0]")