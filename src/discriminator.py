import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*24*32, 100),
            nn.Tanh(),
            nn.Linear(100, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print('after convolution =', x.size())
        x = x.view(-1, 64 * 24 * 32)
        x = self.fc(x)
        # print('after fc = ', x.size())
        return x


'''x = torch.rand([5, 4, 192, 256])
model = Discriminator()
print('Discriminator input', x.size())
out = model(x)
print('Discriminator out ', out.size())'''
