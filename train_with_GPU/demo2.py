import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# dataset
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# model define
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

if __name__=="__main__":
    # Parameters
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    dataset = RandomDataset(input_size, data_size)
    # dataloader define
    rand_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size, shuffle=True)

    # model init
    model = Model(input_size, output_size)

    # cuda devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move model to device 
    model.to(device)

    for data in rand_loader:
        # move data to device 
        input = data.to(device)
        output = model(input)

        # compute loss

        # backward

        # update
        
        print("Outside: input size", input.size(),
            "output_size", output.size())

    torch.save(model.module.cpu().state_dict(), "model.pth")