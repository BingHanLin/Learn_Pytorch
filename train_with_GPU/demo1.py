import torch
import math

data_type = torch.float

# compute on cpu or gpu
device = torch.device("cpu")
# device = torch.device("cuda:0")

x = torch.linspace(-math.pi, math.pi, 1500, device=device,
                   dtype=data_type)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=data_type, requires_grad=True)
b = torch.randn((), device=device, dtype=data_type, requires_grad=True)
c = torch.randn((), device=device, dtype=data_type, requires_grad=True)
d = torch.randn((), device=device, dtype=data_type, requires_grad=True)

learning_rate = 1e-6
for i in range(2):

    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = (y_pred - y).pow(2).sum()
    if i % 500 == 0:
        print(i, loss.item())

    # loss.backward(retain_graph=True)
    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()
    d.grad.zero_()

print(
    f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
