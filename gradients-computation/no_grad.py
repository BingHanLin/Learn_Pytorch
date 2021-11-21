import torch

if __name__ == '__main__':
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([2.0])

    with torch.no_grad():  # auto grad is close in this section
        z_no_grad = w*x + b

    z_no_grad.backward()  # error: does not require grad and does not have a grad_fn
