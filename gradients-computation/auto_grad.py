import torch

if __name__ == '__main__':
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([2.0])

    z = w*x + b

    z.backward()

    print('w.grad : {}'.format(w.grad))  # = x
    print('b.grad : {}'.format(b.grad))  # = 1.0
