import torch

if __name__ == '__main__':
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([2.0])

    z1 = w*x + b
    z2 = w*x + b

    z1.backward()
    z2.backward()

    print('w.grad is accumulated: {}'.format(w.grad))  # = x + x
    print('b.grad is accumulated: {}'.format(b.grad))  # = 1.0 + 1.0

    # reset grad to zero
    w.grad.zero_()
    b.grad.zero_()

    print('w.grad is reset to zero: {}'.format(w.grad))  # = 0.0
    print('b.grad is reset to zero: {}'.format(b.grad))  # = 0.0

    z3 = w*x + b
    z3.backward()

    print('w.grad is correct now: {}'.format(w.grad))  # = 0.0
    print('b.grad is correct now: {}'.format(b.grad))  # = 0.0
