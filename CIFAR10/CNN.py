import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, batch_size):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.3),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64*8*8, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(100, 10)

        # torch.nn.init.xavier_normal_(self.fc.weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        out = self.fc(out)

        return out