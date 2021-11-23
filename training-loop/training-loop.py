import torch
import torch.optim as optim


def model(input, w, b):
    return w * input + b


def loss_fun(result, truth):
    return ((result-truth)**2).mean()


def training_loop(epochs_number, optimizer, params, input, truth):
    for epoch in range(1, epochs_number+1):
        result = model(input, *params)
        loss = loss_fun(result, truth)

        optimizer.zero_grad()  # set grads to zero
        loss.backward()  # compute grads
        optimizer.step()  # update params

        if epoch % 500 == 0:
            print("Grads: {}.".format(params.grad))
            print("Epoch {}, Loss {}.".format(epoch, float(loss)))


if __name__ == '__main__':

    truth = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    input = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

    truth = torch.tensor(truth)
    input = 0.1*torch.tensor(input)

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    optimizer = optim.SGD([params], lr=1e-2)
    training_loop(5000, optimizer, params, input, truth)

    print(params)
